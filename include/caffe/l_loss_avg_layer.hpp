#ifndef CAFFE_L_LOSS_AVG_LAYERS_HPP_
#define CAFFE_L_LOSS_AVG_LAYERS_HPP_

#include "caffe/blob.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
class LLossAVGLayer : public LossLayer<Dtype> {
public:
  explicit LLossAVGLayer(const LayerParameter& param) 
    : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
        
  virtual inline const char* type() const { return "LLossAVG"; } 
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
        
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, 
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, 
                            const vector<Blob<Dtype>*>& bottom);

private:
  int clip_size_, batch_size_, instance_size_, feature_size_;
  bool normalize_;
  Dtype margin_width_;
  Blob<Dtype> S_video_; // per video similarity for @f$ (i, j) @f$
  Blob<Dtype> I_video_; // per video label sim indicators
  Blob<Dtype> video_centroid_;
  Blob<Dtype> pairwise_loss_;
  Blob<Dtype> accuracy_;
};
  
} // namespace caffe

#endif // CAFFE_L_LOSS_LAYERS_HPP_
