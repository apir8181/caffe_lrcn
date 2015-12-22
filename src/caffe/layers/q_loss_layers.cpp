
#include "caffe/q_loss_layers.hpp"

namespace caffe {
  
template <typename Dtype>
void QLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  normalize_ = this->layer_param_.loss_param().normalize();
  axis_ = this->layer_param_.q_loss_param().axis();
  scale_ = this->layer_param_.q_loss_param().scale();
}

template <typename Dtype>
void QLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> shape(0);
  top[0]->Reshape(shape);
  temp_.ReshapeLike(*bottom[0]);
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
