
#include "caffe/q_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <stdio.h>

namespace caffe {

//Q loss margin
template <typename Dtype>
__global__ void kernel_q_loss(
  const Dtype* data, Dtype* out, 
  const int count, const float margin) {

  CUDA_KERNEL_LOOP(i, count) {
    Dtype pre_loss = margin - fabs(data[i]);
    out[i] = (pre_loss > 0) ? pre_loss : 0;
  }

}

template <typename Dtype>
__global__ void kernel_q_backprop(
  const Dtype* loss, const Dtype* data, Dtype* out, const int count) {

  CUDA_KERNEL_LOOP(i, count) {
    Dtype sign = data[i] > 0 ? 1 : -1;
    out[i] = loss[i] > 0 ? -sign : 0; 
  }

}


template <typename Dtype>
void QLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int bottom_data_count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bitwise_loss = loss_.mutable_gpu_data();
  kernel_q_loss<Dtype><<<CAFFE_GET_BLOCKS(bottom_data_count),
    CAFFE_CUDA_NUM_THREADS>>>(bottom_data, bitwise_loss, 
                              bottom_data_count, margin_width_);

  Dtype loss;
  caffe_gpu_asum<Dtype>(bottom_data_count, bitwise_loss, &loss);
  loss *= lambda_ / bottom_data_count;

  top[0]->mutable_cpu_data()[0] = loss;

  // for (int i = 0; i < bottom_data_count; ++ i)
  //    LOG(INFO) << i << " data:" << bottom[0]->cpu_data()[i]
  //              << " loss:" << loss_.cpu_data()[i];
}


template <typename Dtype>
void QLossLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, 
  const vector<Blob<Dtype>*>& bottom) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bitwise_loss = loss_.mutable_gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int bottom_data_count = bottom[0]->count();

  kernel_q_backprop<Dtype><<<CAFFE_GET_BLOCKS(bottom_data_count),
    CAFFE_CUDA_NUM_THREADS>>>(bitwise_loss, bottom_data, bottom_diff, bottom_data_count);
  caffe_gpu_scal<Dtype>(bottom_data_count, lambda_ / bottom_data_count, bottom_diff);
}


INSTANTIATE_LAYER_GPU_FUNCS(QLossLayer);

} // namespace caffe
