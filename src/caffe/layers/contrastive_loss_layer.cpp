#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ContrastiveLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  //printf("Entering ContrastiveLossLayer LayerSetUp method \n");
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
}

/*
template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      diff_.mutable_cpu_data());  // a_i-b_i
  const int channels = bottom[0]->channels();
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    dist_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        diff_.cpu_data() + (i*channels), diff_.cpu_data() + (i*channels));
    if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
      loss += dist_sq_.cpu_data()[i];
    } else {  // dissimilar pairs
      loss += std::max(margin-dist_sq_.cpu_data()[i], Dtype(0.0));
    }
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  (*top)[0]->mutable_cpu_data()[0] = loss;
}
*/



template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      diff_.mutable_cpu_data());  // a_i-b_i
  const int channels = bottom[0]->channels();
  /*
   * margin refers to the maximum value of energy -- parameter Q in the paper
   */

   printf("CLL : the values of a_i are \n");
    for (int i = 0; i < bottom[0]->num(); ++i) {
       for (int j = 0; j < channels; ++j) {
          printf("%f \t ",(float) bottom[0]->cpu_data()[i*channels+j] );
      }
    }
   printf("CLL : End printing values of a_i\n");

  printf("CLL : the values of b_i are \n");
    for (int i = 0; i < bottom[1]->num(); ++i) {
       for (int j = 0; j < channels; ++j) {
          printf("%f \t ",(float) bottom[1]->cpu_data()[i*channels+j] );
      }
    }
   printf("CLL : End printing values of b_i\n");

   printf("CLL : the diff values for the input vector are \n");
   for(int temp = 0 ; temp < count ; temp++){
    printf("%f \t ",(float) diff_.mutable_cpu_data()[temp] );
   }
   printf("CLL : End printing the diff values\n");

  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  //margin = Dtype(1000);
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    dist_sq_.mutable_cpu_data()[i] = caffe_cpu_asum(channels,
        diff_.cpu_data() + (i*channels));

   printf("CLL : values of L1 norm are , %f \n", (float) dist_sq_.mutable_cpu_data()[i] );
    /* 
     * 1 is similar pair, 0 is impostor pair.
     * The paper follows opposite notation
     */

    printf("CLL: label : %d \n", bottom[2]->cpu_data()[i]);
    if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
      
      loss += Dtype(2) / margin * dist_sq_.cpu_data()[i] * dist_sq_.cpu_data()[i];

      printf(" CLL: loss computed : %f\n", dist_sq_.cpu_data()[i]);
    
    } else {  // dissimilar pairs
      //loss += std::max(margin-dist_sq_.cpu_data()[i], Dtype(0.0));
      printf("CLL : the exponent of 1 is : %f \n",exp(Dtype(1)));
      printf("CLL : the exponent of -1 is : %f \n", exp(Dtype(-1)));
      loss += Dtype(2) * margin * exp(-Dtype(2.77) / margin * dist_sq_.cpu_data()[i]);
       printf(" CLL: loss computed : %f\n", dist_sq_.cpu_data()[i]);
    }
    printf("CLL: value of label : %d \n", static_cast<int>(bottom[2]->cpu_data()[i]));
    printf("CLL: value of margin : %f \n", (float) margin);

  }
  loss = loss / static_cast<Dtype>(bottom[0]->num());
  printf("CLL: value of loss : %f \n", loss);
  (*top)[0]->mutable_cpu_data()[0] = loss;
}

/*
template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>((*bottom)[i]->num());
      int num = (*bottom)[i]->num();
      int channels = (*bottom)[i]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = (*bottom)[i]->mutable_cpu_diff();
        if (static_cast<int>((*bottom)[2]->cpu_data()[j])) {  // similar pairs
          caffe_cpu_axpby(
              channels,
              alpha,
              diff_.cpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
        } else {  // dissimilar pairs
          if ((margin-dist_sq_.cpu_data()[j]) > Dtype(0.0)) {
            caffe_cpu_axpby(
                channels,
                -alpha,
                diff_.cpu_data() + (j*channels),
                Dtype(0.0),
                bout + (j*channels));
          } else {
            caffe_set(channels, Dtype(0), bout + (j*channels));
          }
        }
      }
    }
  }
}
*/
template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  //margin = Dtype(1000);
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>((*bottom)[i]->num());
         // printf("CLL:value of alpha is %f \n", (float)alpha);
         // printf("CLL:value of CPU diff is %f \n", (float) top[0]->cpu_diff()[0]);
         // printf("CLL:value of bottom num is %d \n", (int) (*bottom)[i]->num());
      int num = (*bottom)[i]->num();
      int channels = (*bottom)[i]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = (*bottom)[i]->mutable_cpu_diff();
        if (static_cast<int>((*bottom)[2]->cpu_data()[j])) {  // similar pairs
          for(int k = 0 ; k < channels ; k ++){
            Dtype gradient_sign = diff_.cpu_data()[(j*channels) + k] > 0 ? 1 : -1;
            bout[(j*channels) + k] += alpha * dist_sq_.mutable_cpu_data()[j] 
                                    * gradient_sign * 4 / margin;
          }
          /*
          caffe_cpu_axpby(
              channels,
              alpha,
              diff_.cpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
          */
        } else {  // dissimilar pairs
          
          for(int k = 0 ; k < channels ; k ++){
            Dtype gradient_sign = diff_.cpu_data()[(j*channels) + k] > 0 ? 1 : -1;
            bout[(j*channels) + k] += alpha * Dtype(2) * -Dtype(2.77) 
                                    * exp(-Dtype(2.77) / margin * dist_sq_.mutable_cpu_data()[j] )
                                    * gradient_sign;
          }

          /*
          if ((margin-dist_sq_.cpu_data()[j]) > Dtype(0.0)) {
            caffe_cpu_axpby(
                channels,
                -alpha,
                diff_.cpu_data() + (j*channels),
                Dtype(0.0),
                bout + (j*channels));
          } else {
            caffe_set(channels, Dtype(0), bout + (j*channels));
          }
          */
        }
      }
    }
  }
/*
  // print values for debugging 
  for (int i = 0; i < 2; ++i) {
    int num = (*bottom)[i]->num();
      int channels = (*bottom)[i]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = (*bottom)[i]->mutable_cpu_diff();
       // if (static_cast<int>((*bottom)[2]->cpu_data()[j])) {  // similar pairs
          for(int k = 0 ; k < channels ; k ++){
            printf("CLL: Sample Num : %d, isSimilarPair : %d \n",j,static_cast<int>((*bottom)[2]->cpu_data()[j]));
            printf("CLL: Bottom Blob Num : %d , Channel Num : %d , Gradient : %f \n", i , k, (float) bout[(j*channels) + k]);
          }
      }
  }
  */

}

#ifdef CPU_ONLY
STUB_GPU(ContrastiveLossLayer);
#endif

INSTANTIATE_CLASS(ContrastiveLossLayer);

}  // namespace caffe
