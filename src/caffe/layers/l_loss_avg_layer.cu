
#include "caffe/l_loss_avg_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_get_centroid(
    const Dtype* bottom_data, Dtype* out,
    const int clip_size, const int batch_size, const int feature_size) {

  CUDA_KERNEL_LOOP(id, batch_size * feature_size) {
    int i = id / feature_size, d = id % feature_size;
    Dtype feature_val = 0;
    for (int t = 0; t < clip_size; ++ t) {
      int idx = (t * batch_size + i) * feature_size + d;
      feature_val += bottom_data[idx] / clip_size;
    }
    out[id] = feature_val;
  }

}


template <typename Dtype>
__global__ void kernel_centroid_length(
  const Dtype* centroid, Dtype* out, 
  const int batch_size, const int feature_size) {

  CUDA_KERNEL_LOOP(i, batch_size) {
    Dtype val = 0;
    for (int d = 0; d < feature_size; ++ d) {
      int id = i * feature_size + d;
      val += centroid[id] * centroid[id];
    }
    out[i] = sqrt(val);
  }

}


template <typename Dtype>
__global__ void kernel_video_similarity(
    const Dtype* video_centroid, const Dtype* centroid_length, Dtype* out, 
    const int batch_size, const int feature_size) {
  
  CUDA_KERNEL_LOOP(ij, batch_size * batch_size) {
    int i = ij / batch_size, j = ij % batch_size;
    Dtype val = 0;
    for (int d = 0; d < feature_size; ++ d) {
      int id = i * feature_size + d, jd = j * feature_size + d;
      val += video_centroid[id] * video_centroid[jd];
    }
    //val /= centroid_length[i] * centroid_length[j];
    val /= feature_size;
    out[ij] = val;
  }

}


template <typename Dtype>
__global__ void kernel_label_indicator(
  const Dtype* label, Dtype* out, const int batch_size) {

  CUDA_KERNEL_LOOP(ij, batch_size * batch_size) {
    int i = ij / batch_size, j = ij % batch_size;
    int label1 = (int) label[i];
    int label2 = (int) label[j];
    bool has_common = (label1 & label2) > 0;
    out[ij] = has_common ? 1 : -1;
  }

}

template <typename Dtype>
__global__ void kernel_pairwise_error(
    const Dtype* S_video, const Dtype* I_video,
    Dtype* out, const int count) {

   CUDA_KERNEL_LOOP(i, count) {
     Dtype pre_loss = I_video[i] - S_video[i];
     out[i] = pre_loss;
   }

}


template <typename Dtype>
__global__ void kernel_pairwise_loss(
    const Dtype* S_video, const Dtype* I_video,
    Dtype* out, const int count) {

   CUDA_KERNEL_LOOP(i, count) {
     Dtype pre_loss = I_video[i] - S_video[i];
     out[i] = pre_loss * pre_loss;
   }

}


template <typename Dtype>
void LLossAVGLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  Dtype* video_centroid = video_centroid_.mutable_gpu_data();
  Dtype* centroid_length = centroid_length_.mutable_gpu_data();
  Dtype* S_video = S_video_.mutable_gpu_data();
  Dtype* I_video = I_video_.mutable_gpu_data();
  Dtype* pairwise_error = pairwise_error_.mutable_gpu_data();
  Dtype* pairwise_loss = pairwise_loss_.mutable_gpu_data();

  // calculate video centroid
  kernel_get_centroid<Dtype><<<CAFFE_GET_BLOCKS(batch_size_ * feature_size_),
    CAFFE_CUDA_NUM_THREADS>>>(bottom_data, video_centroid,
                              clip_size_, batch_size_, feature_size_);

  kernel_centroid_length<Dtype><<<CAFFE_GET_BLOCKS(batch_size_),
    CAFFE_CUDA_NUM_THREADS>>>(video_centroid, centroid_length, 
                              batch_size_, feature_size_);

  // calculate video pair similarity
  kernel_video_similarity<Dtype><<<CAFFE_GET_BLOCKS(batch_size_ * batch_size_),
    CAFFE_CUDA_NUM_THREADS>>>(video_centroid, centroid_length, S_video, 
                              batch_size_, feature_size_);

  // calculate video label indicator
  kernel_label_indicator<Dtype><<<CAFFE_GET_BLOCKS(batch_size_ * batch_size_),
    CAFFE_CUDA_NUM_THREADS>>>(bottom_label, I_video, batch_size_);

  // calculate pairwise error
  kernel_pairwise_error<Dtype><<<CAFFE_GET_BLOCKS(batch_size_ * batch_size_),
    CAFFE_CUDA_NUM_THREADS>>>(S_video, I_video, pairwise_error,
                              batch_size_ * batch_size_);
  kernel_pairwise_loss<Dtype><<<CAFFE_GET_BLOCKS(batch_size_ * batch_size_),
    CAFFE_CUDA_NUM_THREADS>>>(S_video, I_video, pairwise_loss,
                              batch_size_ * batch_size_);

  Dtype loss, abs_error;
  caffe_gpu_asum<Dtype>(batch_size_ * batch_size_, pairwise_loss, &loss);
  caffe_gpu_asum<Dtype>(batch_size_ * batch_size_, pairwise_error, &abs_error);
  if (normalize_) {
    loss /= batch_size_ * batch_size_;
    abs_error /= batch_size_ * batch_size_;
  }
  LOG(INFO) << "loss:" << loss;
  LOG(INFO) << "abs_error:" << abs_error;

  top[0]->mutable_cpu_data()[0] = loss;

  // for (int i = 0; i < batch_size_ * batch_size_; ++ i) {
  //   LOG(INFO) << " ij:" << i
  //             << " loss:" << pairwise_loss_.mutable_cpu_data()[i]
  //             << " label:" << I_video_.mutable_cpu_data()[i]
  //             << " sim:" << S_video_.mutable_cpu_data()[i];
  // }

  // for (int i = 0; i < bottom[1]->count(); ++ i) {
  //    LOG(INFO) << "label: " << i << " " << bottom[1]->cpu_data()[i];
  // }
  //  for (int i = 0; i < I_video_.count(); ++ i) {
  //    LOG(INFO) << "video label similarity: " << i << " " << I_video_.cpu_data()[i];
  // }
  // for (int i = 0; i < S_video_.count(); ++ i) {
  //   LOG(INFO) << "video similarity: " << i << " " << S_video_.cpu_data()[i];
  // }
  //LOG(INFO) << "bias " << this->blobs_[0]->cpu_data()[0];
}

template <typename Dtype>
__global__ void kernel_centroid_backprop(
    const Dtype* pairwise_error, const Dtype* video_centroid, 
    const Dtype* centroid_length, const Dtype* video_sim, Dtype* out,
    const int clip_size, const int batch_size, const int feature_size) {

  CUDA_KERNEL_LOOP(id, batch_size * feature_size) {
    int i = id / feature_size, d = id % feature_size;
    Dtype val = 0;
    for (int j = 0; j < batch_size; ++ j) {
      int ij = i * batch_size + j;
      int jd = j * feature_size + d;
      Dtype weight = -2 * pairwise_error[ij];
      // Dtype l_part = video_centroid[jd] / (centroid_length[i] * centroid_length[j]);
      // Dtype r_part = video_centroid[id] * video_sim[ij] / powf(centroid_length[i], 2);
      // val += weight * (l_part - r_part);
      if (i != j) {
        val += weight * video_centroid[jd];
      } else {
        val += weight * video_centroid[id] * 2;
      }
    }
    out[id] = val / feature_size;
  }

}


template <typename Dtype>
__global__ void kernel_backprop(
    const Dtype* centroid_diff, Dtype* out, 
    int clip_size, int batch_size, int feature_size) {
  
  CUDA_KERNEL_LOOP(ti, clip_size * batch_size) {
    int i = ti % batch_size;
    for (int d = 0; d < feature_size; ++ d) {
      out[ti*feature_size+d] = centroid_diff[i*feature_size+d] / clip_size;
    }
  }

}


template <typename Dtype>
void LLossAVGLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, 
  const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const int bottom_count = bottom[0]->count();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* pairwise_error = pairwise_error_.gpu_data();
    const Dtype* video_centroid = video_centroid_.gpu_data();
    const Dtype* centroid_length = centroid_length_.gpu_data();
    const Dtype* S_video = S_video_.gpu_data();
    Dtype* centroid_diff = centroid_diff_.mutable_gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    // backprop residue to bottom
    kernel_centroid_backprop<Dtype><<<CAFFE_GET_BLOCKS(batch_size_ * feature_size_),
      CAFFE_CUDA_NUM_THREADS>>>(pairwise_error, video_centroid, 
                                centroid_length, S_video, centroid_diff,
                                clip_size_, batch_size_, feature_size_);

    kernel_backprop<Dtype><<<CAFFE_GET_BLOCKS(clip_size_ * batch_size_),
      CAFFE_CUDA_NUM_THREADS>>>(centroid_diff, bottom_diff,
                                clip_size_, batch_size_, feature_size_);

    if (normalize_) {
         Dtype scale_factor = 1.0 / (batch_size_ * batch_size_);
         caffe_gpu_scal<Dtype>(bottom[0]->count(), scale_factor, bottom_diff);
    } 

    Dtype bottom_diff_som, bottom_data_som;
    caffe_gpu_asum<Dtype>(bottom_count, bottom_data, &bottom_data_som);
    caffe_gpu_asum<Dtype>(bottom_count, bottom_diff, &bottom_diff_som);
    bottom_data_som /= bottom_count;
    bottom_diff_som /= bottom_count;
    LOG(INFO) << "bottom data SOM:" << bottom_data_som;
    LOG(INFO) << "bottom diff SOM:" << bottom_diff_som;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(LLossAVGLayer);
} // namespace caffe
