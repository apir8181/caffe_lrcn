
#include "caffe/l_loss_outer_cosine_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <stdio.h>

namespace caffe {

template <typename Dtype>
__global__ void kernel_frame_length_cosine(
  const Dtype* data, Dtype* out, 
  const int instance_size, const int feature_size) {

  CUDA_KERNEL_LOOP(ti, instance_size) {
    Dtype val = 0;
    for (int d = 0; d < feature_size; ++ d) {
      int idx = ti * feature_size + d;
      val += data[idx] * data[idx];
    }
    out[ti] = sqrt(val);
  }

}


template <typename Dtype>
__global__ void kernel_frame_similarity_cosine(
  const Dtype* data, const Dtype* data_length, Dtype* out, 
  const int instance_size, const int feature_size) {
  
  CUDA_KERNEL_LOOP(ij, instance_size * instance_size) {
    Dtype val = 0;
    int i = ij / instance_size, j = ij % instance_size;
    for (int d = 0; d < feature_size; ++ d) {
      int id = i * feature_size + d, jd = j * feature_size + d;
      val += data[id] * data[jd];
    }
    out[ij] = val / (data_length[i] * data_length[j]);
  }

}


template <typename Dtype>
__global__ void kernel_frame_indicator_cosine(
  const Dtype* label, Dtype* out, const int instance_size) {
  
  CUDA_KERNEL_LOOP(ij, instance_size * instance_size) {
    int i = ij / instance_size, j = ij % instance_size;
    int label1 = (int) label[i], label2 = (int) label[j];
    bool matched = (label1 & label2) > 0;
    out[ij] = matched ? 1 : -1;
  }

}


template <typename Dtype>
__global__ void kernel_pairwise_error_cosine(
    const Dtype* S_frame, const Dtype* I_frame,
    Dtype* out, const int count) {

   CUDA_KERNEL_LOOP(i, count) {
     Dtype pre_loss = I_frame[i] - S_frame[i];
     out[i] = pre_loss;
   }

}


template <typename Dtype>
__global__ void kernel_pairwise_loss_cosine(
    const Dtype* S_frame, const Dtype* I_frame,
    Dtype* out, const int count) {

   CUDA_KERNEL_LOOP(i, count) {
     Dtype pre_loss = I_frame[i] - S_frame[i];
     out[i] = pre_loss * pre_loss;
   }

}


template <typename Dtype>
void LLossOuterCosineLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  Dtype* L_frame = L_frame_.mutable_gpu_data();
  Dtype* S_frame = S_frame_.mutable_gpu_data();
  Dtype* I_frame = I_frame_.mutable_gpu_data();
  Dtype* pairwise_error = pairwise_error_.mutable_gpu_data();
  Dtype* pairwise_loss = pairwise_loss_.mutable_gpu_data();
  int pair_size = instance_size_ * instance_size_;

  kernel_frame_length_cosine<Dtype><<<CAFFE_GET_BLOCKS(instance_size_),
    CAFFE_CUDA_NUM_THREADS>>>(bottom_data, L_frame, instance_size_, feature_size_);

  // calculate video pair similarity
  kernel_frame_similarity_cosine<Dtype><<<CAFFE_GET_BLOCKS(pair_size),
    CAFFE_CUDA_NUM_THREADS>>>(bottom_data, L_frame, S_frame, 
                              instance_size_, feature_size_);

  // calculate video label indicator
  kernel_frame_indicator_cosine<Dtype><<<CAFFE_GET_BLOCKS(pair_size),
    CAFFE_CUDA_NUM_THREADS>>>(bottom_label, I_frame, instance_size_);

  // calculate pairwise error
  kernel_pairwise_error_cosine<Dtype><<<CAFFE_GET_BLOCKS(pair_size),
    CAFFE_CUDA_NUM_THREADS>>>(S_frame, I_frame, pairwise_error, pair_size);
                              
  kernel_pairwise_loss_cosine<Dtype><<<CAFFE_GET_BLOCKS(pair_size),
    CAFFE_CUDA_NUM_THREADS>>>(S_frame, I_frame, pairwise_loss, pair_size);

  Dtype loss, abs_error;
  caffe_gpu_asum<Dtype>(pair_size, pairwise_loss, &loss);
  caffe_gpu_asum<Dtype>(pair_size, pairwise_error, &abs_error);
  if (normalize_) {
    loss /= pair_size;
    abs_error /= pair_size;
  }
  LOG(INFO) << "loss:" << loss;
  LOG(INFO) << "abs_error:" << abs_error;

  top[0]->mutable_cpu_data()[0] = loss;

  // for (int i = 0; i < pair_size; ++ i) {
  //   LOG(INFO) << " ij:" << i
  //             << " loss:" << pairwise_loss_.mutable_cpu_data()[i]
  //             << " label:" << I_frame_.mutable_cpu_data()[i]
  //             << " sim:" << S_frame_.mutable_cpu_data()[i];
  // }
}

template <typename Dtype>
__global__ void kernel_backprop_cosine(
  const Dtype* pairwise_error, const Dtype* data, 
  const Dtype* data_length, const Dtype* S_frame, Dtype* out,
  const int instance_size, int feature_size) {

  CUDA_KERNEL_LOOP(id, instance_size * feature_size) {
    int i = id / feature_size, d = id % feature_size;
    Dtype val = 0;
    for (int j = 0; j < instance_size; ++ j) {
      int ij = i * instance_size + j;
      int jd = j * feature_size + d;
      Dtype weight = -2 * pairwise_error[ij];
      Dtype l_part = data[jd] / (data_length[i] * data_length[j]);
      Dtype r_part = data[id] * S_frame[ij] / powf(data_length[i], 2);
      val += weight * (l_part - r_part);
    }
    out[id] = val;
  }
}


template <typename Dtype>
void LLossOuterCosineLayer<Dtype>::Backward_gpu(
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
    const Dtype* data_length = L_frame_.gpu_data();
    const Dtype* S_frame = S_frame_.gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    kernel_backprop_cosine<Dtype><<<CAFFE_GET_BLOCKS(instance_size_ * feature_size_),
      CAFFE_CUDA_NUM_THREADS>>>(pairwise_error, bottom_data, 
                                data_length, S_frame, bottom_diff,
                                instance_size_, feature_size_);

    if (normalize_) {
      Dtype scale_factor = 1.0 / (instance_size_ * instance_size_);
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

INSTANTIATE_LAYER_GPU_FUNCS(LLossOuterCosineLayer);
} // namespace caffe
