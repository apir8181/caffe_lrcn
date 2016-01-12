
#include "caffe/q_loss_layer.hpp"

namespace caffe {
  
template <typename Dtype>
void QLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  margin_width_ = this->layer_param_.q_loss_param().margin_width();
  lambda_ = this->layer_param_.q_loss_param().lambda();
}

template <typename Dtype>
void QLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> single_shape(0);
  top[0]->Reshape(single_shape);

  vector<int> shape(1, bottom[0]->count());
  loss_.Reshape(shape);
}

template <typename Dtype>
void QLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void QLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(QLossLayer);
#endif

INSTANTIATE_CLASS(QLossLayer);
REGISTER_LAYER_CLASS(QLoss);

}
