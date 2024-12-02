#include "yolov11.h"
#include <cpu.h>
#include <iostream>
#include <vector>


// 快速指数
static float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}


static float softmax(
	const float* src,
	float* dst,
	int length
)
{
	float alpha = -FLT_MAX;
	for (int c = 0; c < length; c++)
	{
		float score = src[c];
		if (score > alpha)
		{
			alpha = score;
		}
	}

	float denominator = 0;
	float dis_sum = 0;
	for (int i = 0; i < length; ++i)
	{
		// dst[i] = expf(src[i] - alpha);
		dst[i] = fast_exp(src[i] - alpha); // // 使用 fast_exp 代替 expf
		
		denominator += dst[i];
	}
	for (int i = 0; i < length; ++i)
	{
		dst[i] /= denominator;
		dis_sum += i * dst[i];
	}
	return dis_sum;
}

static void generate_proposals(
        int stride,
        const ncnn::Mat& feat_blob,
        const float prob_threshold,
        std::vector<Object>& objects
)
{
    const int reg_max = 16;
    float dst[16];
    const int num_w = feat_blob.w;
    const int num_grid_y = feat_blob.c;
    const int num_grid_x = feat_blob.h;

    const int num_class = num_w - 4 * reg_max;

    for (int i = 0; i < num_grid_y; i++)
    {
        for (int j = 0; j < num_grid_x; j++)
        {
            const float* matat = feat_blob.channel(i).row(j);

            int class_index = 0;
			float class_score = -FLT_MAX;
			for (int c = 0; c < num_class; c++)
			{
				float score = matat[4 * reg_max + c];
				if (score > class_score)
				{
					class_index = c;
					class_score = score;
				}
			}

            if (class_score >= prob_threshold)
			{

				float x0 = j + 0.5f - softmax(matat, dst, 16);
				float y0 = i + 0.5f - softmax(matat + 16, dst, 16);
				float x1 = j + 0.5f + softmax(matat + 2 * 16, dst, 16);
				float y1 = i + 0.5f + softmax(matat + 3 * 16, dst, 16);

				x0 *= stride;
				y0 *= stride;
				x1 *= stride;
				y1 *= stride;

				Object obj;
				obj.rect.x = x0;
				obj.rect.y = y0;
				obj.rect.width = x1 - x0;
				obj.rect.height = y1 - y0;
				obj.label = class_index;
				obj.prob = class_score;
				objects.push_back(obj);

			}
           
        }
    }

}

static float clamp(
        float val,
        float min = 0.f,
        float max = 1280.f
)
{
    return val > min ? (val < max ? val : max) : min;
}

// 计算 IoU (Intersection over Union) 的函数
static float compute_iou(const cv::Rect& box1, const cv::Rect& box2)
{
    float inter_area = (box1 & box2).area();
    float union_area = box1.area() + box2.area() - inter_area;
    return inter_area / union_area;
}


static void non_max_suppression(
        std::vector<Object>& proposals,
        std::vector<Object>& results,
        int orin_h,
        int orin_w,
        float dh = 0,
        float dw = 0,
        float ratio_h = 1.0f,
        float ratio_w = 1.0f,
        float conf_thres = 0.25f,
        float iou_thres = 0.65f
)
{
    results.clear();
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;

    // 1. 将 proposals 中的矩形框和分数提取出来
    for (auto& pro : proposals)
    {
        bboxes.push_back(pro.rect);
        scores.push_back(pro.prob);
        labels.push_back(pro.label);
    }

    // 2. 按照分数进行排序，排序后从高到低进行 NMS
    std::vector<int> sorted_indices(scores.size());
    for (size_t i = 0; i < scores.size(); ++i)
    {
        sorted_indices[i] = i;
    }

    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&scores](int i1, int i2) { return scores[i1] > scores[i2]; });

    // 3. 执行非最大抑制
    std::vector<bool> keep(scores.size(), true);
    for (size_t i = 0; i < sorted_indices.size(); ++i)
    {
        if (!keep[sorted_indices[i]]) continue;

        const auto& box_i = bboxes[sorted_indices[i]];
        float score_i = scores[sorted_indices[i]];

        if (score_i < conf_thres)
            break;

        for (size_t j = i + 1; j < sorted_indices.size(); ++j)
        {
            if (!keep[sorted_indices[j]]) continue;

            const auto& box_j = bboxes[sorted_indices[j]];
            float iou = compute_iou(box_i, box_j);

            // 如果 IOU 大于阈值，则将 box_j 去除
            if (iou > iou_thres)
            {
                keep[sorted_indices[j]] = false;
            }
        }
    }

    // 4. 将满足条件的框添加到结果中
    for (size_t i = 0; i < sorted_indices.size(); ++i)
    {
        if (keep[sorted_indices[i]])
        {
            const auto& bbox = bboxes[sorted_indices[i]];
            float score = scores[sorted_indices[i]];
            int label = labels[sorted_indices[i]];

            float x0 = bbox.x;
            float y0 = bbox.y;
            float x1 = bbox.x + bbox.width;
            float y1 = bbox.y + bbox.height;

            x0 = (x0 - dw) / ratio_w;
            y0 = (y0 - dh) / ratio_h;
            x1 = (x1 - dw) / ratio_w;
            y1 = (y1 - dh) / ratio_h;

            x0 = clamp(x0, 0.f, orin_w);
            y0 = clamp(y0, 0.f, orin_h);
            x1 = clamp(x1, 0.f, orin_w);
            y1 = clamp(y1, 0.f, orin_h);

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.prob = score;
            obj.label = label;

            results.push_back(obj);
        }
    }
}

Inference::Inference(){
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int Inference::loadNcnnNetwork(AAssetManager* mgr, const char* modeltype , const int& modelInputShape, const float* meanVals, const float* normVals, bool useGpu)
{
    target_size = modelInputShape;
    gpuEnabled = useGpu;

    net.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    net.opt = ncnn::Option();

#if NCNN_VULKAN
    net.opt.use_vulkan_compute = useGpu;
#endif

    net.opt.num_threads = ncnn::get_big_cpu_count();
    net.opt.blob_allocator = &blob_pool_allocator;
    net.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "yolov11%s_ncnn_model/yolov11n.ncnn.param", modeltype);
    sprintf(modelpath, "yolov11%s_ncnn_model/yolov11n.ncnn.bin", modeltype);

    net.load_param(mgr, parampath);
    net.load_model(mgr, modelpath);

    this->meanVals[0] = meanVals[0];
    this->meanVals[1] = meanVals[1];
    this->meanVals[2] = meanVals[2];
    this->normVals[0] = normVals[0];
    this->normVals[1] = normVals[1];
    this->normVals[2] = normVals[2];
    return 0;
}

std::vector<Object> Inference::runInference(const cv::Mat &bgr)
{
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    int w = img_w;
    int h = img_h;

    float scale = 1.f;
    if (w > h) {
        scale = (float)target_size / w;
        w = target_size;
        h = (int)(h * scale);
    }
    else {
        scale = (float)target_size / h;
        h = target_size;
        w = (int)(w * scale);
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // int wpad = (w + 32 - 1) / 32 * 32 - w;
    // int hpad = (h + 32 - 1) / 32 * 32 - h;

    int wpad = target_size - w;
    int hpad = target_size - h;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2,  wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    in_pad.substract_mean_normalize(meanVals, normVals);

    ncnn::Extractor ex = net.create_extractor();

    ex.input("in0", in_pad);


    std::vector<Object> proposals;

    // stride 8
    {
        ncnn::Mat out;
        ex.extract("out0", out);

        std::vector<Object> objects8;
        generate_proposals(8, out, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat out;
        ex.extract("out1", out);

        std::vector<Object> objects16;
        generate_proposals(16, out, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat out;
        ex.extract("out2", out);

        std::vector<Object> objects32;
        generate_proposals(32, out, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    // objects = proposals;
    std::vector<Object> objects;
    for (auto& pro : proposals)
	{
        float x0 = pro.rect.x;
		float y0 = pro.rect.y;
		float x1 = pro.rect.x + pro.rect.width;
		float y1 = pro.rect.y + pro.rect.height;
		float& score = pro.prob;
		int& label = pro.label;

		x0 = (x0 - (wpad / 2)) / scale;
		y0 = (y0 - (hpad / 2)) / scale;
		x1 = (x1 - (wpad / 2)) / scale;
		y1 = (y1 - (hpad / 2)) / scale;

		x0 = clamp(x0, 0.f, img_w);
		y0 = clamp(y0, 0.f, img_h);
		x1 = clamp(x1, 0.f, img_w);
		y1 = clamp(y1, 0.f, img_h);

		Object obj;
		obj.rect.x = x0;
		obj.rect.y = y0;
		obj.rect.width = x1 - x0;
		obj.rect.height = y1 - y0;
		obj.prob = score;
		obj.label = label;
		objects.push_back(obj);
	}

    non_max_suppression(proposals, objects,
                        img_h, img_w, hpad / 2, wpad / 2,
                        scale, scale, prob_threshold, nms_threshold);


    return objects;

}

int Inference::draw(cv::Mat& bgr, const std::vector<Object>& objects) 
{
    static const char* class_names[] = {
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush"
	};
    cv::Mat res = bgr;
    for (size_t i = 0; i < objects.size(); i++)
	{
		const Object& obj = objects[i];

		fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
			obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

		cv::rectangle(res, obj.rect, cv::Scalar(255, 0, 0));

		char text[256];
		sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

		int x = obj.rect.x;
		int y = obj.rect.y - label_size.height - baseLine;
		if (y < 0)
			y = 0;
		if (x + label_size.width > bgr.cols)
			x = bgr.cols - label_size.width;

		cv::rectangle(res, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
			cv::Scalar(255, 255, 255), -1);

		cv::putText(res, text, cv::Point(x, y + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
	}

    return 0;
}
