**SSM-based RIP Model for Rider Intention Prediction**  
---

#### **1\. Preliminaries**

The backbone of our classification models rely on recent advancements with sequence modelling architectures, particularly state-space models. A recent selective SSM model, namely Mamba, has shown great promise in various different domains, starting off with text, images and most relevant to our task, time-series tasks.  
The task of Rider’s intent prediction could be considered a task for high-dimension time series classification, where the subtle changes in frame embeddings throughout video clips could hint towards the prediction class  
**1.1 Mamba Architecture Overview**  
Mamba is a new SSM architecture built for information-dense data. They are, by design, fully recurrent models that make them suitable for operating on long sequences while being able to selectively prioritize information with the data-driven selection mechanism. They achieve transformer-quality performance while maintaining a linear-time complexity.  
**1.1.1 Mamba Block Architecture**  
The fundamental building block of our model is the Mamba block, which processes input sequences through a series of operations

1. **Input Projection**: The input sequence X of shape (B, V, D) is linearly projected to create two intermediate representations, x and z, both of shape (B, V, ED), where B is the batch size, V is the sequence length, D is the input dimension, and E is an expansion factor.  
1. **Convolutional Layer**: A 1D convolutional operation is applied to x, followed by the SiLU activation function, producing x'.  
1. **Parameter Generation**: The block generates input-dependent parameters A, B, and C through linear projections of x'. Additionally, it computes Δ, which controls the discretization of the continuous-time SSM.  
1. **Selective SSM**: The core of the Mamba block is the selective SSM operation, which processes the input sequence using the generated parameters.  
1. **Output Projection**: The output of the selective SSM is combined with the z intermediate representation and projected back to the original input dimension.

**Algorithm 1: The process of Mamba Block Input**: 𝑿 ∶ (𝐵, 𝑉 , 𝐷)   
**Output**: 𝒀 ∶ (𝐵, 𝑉 , 𝐷)   
1: 𝑥, 𝑧 ∶ (𝐵, 𝑉 , 𝐸𝐷) ← **Linear**(𝑼) {Linear projection}   
2: 𝑥 ′ ∶ (𝐵, 𝑉 , 𝐸𝐷) ← **SiLU**(**Conv1D**(𝑥))   
3: A ∶ (𝐷, 𝑁) ← **𝑃𝑎𝑟𝑎𝑚𝑒𝑡𝑒𝑟** {Structured state matrix}   
4: B,C ∶ (𝐵, 𝑉 , 𝑁) ← **Linear**(𝑥 ′ ), **Linear**(𝑥 ′ )   
5: Δ ∶ (𝐵, 𝑉 , 𝐷) ← **Softplus**(𝑃𝑎𝑟𝑎𝑚𝑒𝑡𝑒𝑟 \+ **Broadcast**(**Linear**(𝑥 ′ )))   
6: **A,B** ∶ (𝐵.𝑉 .𝐷.𝑁) ← **𝑑𝑖𝑠𝑐𝑟𝑒𝑡𝑖𝑧𝑒**(Δ, A,B) {Inputdependent parameters and discretization} 7: 𝑦 ∶ (𝐵, 𝑉 , 𝐸𝐷) ← **SelectiveSSM**(A,B,C)(𝑥 ′ ) 8: 𝑦 ′ ∶ (𝐵, 𝑉 , 𝐸𝐷) ← 𝑦 ⊗ **SiLU**(𝑧)   
9: 𝒀 ∶ (𝐵, 𝑉 , 𝐷) ← **Linear**(𝑦 ′ ) {Linear Projection}  
Where ⊗ denotes element-wise multiplication.  

#### **2\. Methodology**

**2.1 Single View Model**  
**2.1.1 Model Structure**  
Our single-view model for rider intention prediction uses the Mamba SSM architecture to process temporal sequences of frame embeddings extracted from video clips. This approach allows us to capture both short-term and long-term dependencies in the rider's behavior, which are crucial for accurate intention prediction.  
The input to our model consists of VGG16 embeddings extracted from each frame of the frontal view camera footage. Each embedding has a dimension of 512, providing a comprehensive representation of the visual content in each frame.  
To handle the variable-length nature of our video clips, we employ a dynamic padding strategy within each batch during training and inference. This approach allows us to efficiently process sequences of different lengths without losing temporal information or introducing unnecessary computational overhead.  
The architecture of our single-view model is designed to progressively refine the temporal representation of the input sequence. It consists of a series of Mamba blocks, each of which processes the entire sequence and updates its internal state. The number of Mamba blocks is a hyperparameter that we tuned based on empirical performance on the validation set.  
After the sequence has been processed by the Mamba blocks, we apply global average pooling across the temporal dimension. This operation condenses the temporal information into a fixed-size representation, which is crucial for our classification task as it needs to produce a single prediction for the entire sequence.  
The pooled features are then passed through a final linear layer, which maps them to class probabilities corresponding to different rider intentions. We use a softmax activation function to ensure that the output represents a valid probability distribution over the possible intention classes.  
Key hyperparameters that influence our model here are:

* **D\_state:** State dimension of the selective layer  
* **D\_conv:** Kernel size of the convolutional layer in mamba block  
* **Expand:** Expansion factor for internal dimension

**Algorithm 2: Process of Single-View Model**  
**Input**:   
X: (B, T, F) \- Batch of video features  
       B: Batch size  
       T: Sequence length  
       F: Feature dimension (512 for VGG16 features)  
**Output**:  
 Y: (B, C) \- Predicted class probabilities  
 C: Number of classes (rider intentions)  
**Flow**:  
1\. Mamba Block Processing:  
for each Mamba block:  
     x ← **MambaBlock**(x)  
     
2\. Global Pooling:  
**x\_pooled**: (B, D) ← **GlobalAveragePooling**(x)  
     
3\. Classification:  
Y: (B, C) ← **SoftmaxClassifier**(x\_pooled)

**Function SoftmaxClassifie**r(x):  
    logits ← **LinearLayer**(x)  
    probabilities ← **Softmax**(logits)  
    return probabilities

**2.2 Multi-View Model**  
**2.2.1 Model Structure**  
To incorporate information from multiple camera views and potentially improve the accuracy of our predictions, we developed a multi-view model that processes features from the frontal, left side mirror, and right side mirror cameras simultaneously. This approach allows our model to capture a more comprehensive representation of the rider's environment and behavior.  
Our multi-view architecture is based on an ensemble of three single-view Mamba models, each dedicated to processing the features from one of the camera views. This design choice allows each model to specialize in extracting relevant information from its respective view while maintaining the ability to capture view-specific temporal dynamics.  
The multi-view model processes the input features as follows: First, each view's features (frontal, left mirror, and right mirror) are independently fed through a separate Mamba model, as described in the single-view architecture. This parallel processing allows each model to focus on the unique information provided by its respective view.  
After obtaining predictions from each view-specific model, we employ a learnable weighting mechanism to combine these predictions. This approach allows our model to automatically determine the relative importance of each view for the intention prediction task. The weights are initialized randomly and are trained end-to-end with the rest of the model parameters, allowing them to adapt to the specific characteristics of our dataset.  
The final prediction is computed as a weighted sum of the individual view outputs. To ensure that the resulting combination represents a valid probability distribution, we apply a softmax function to the weighted sum. This process can be formalized as:

Y\_combined \= **W\[0\]** \* Y\_front \+ **W\[1\]** \* Y\_left \+ **W\[2\]** \* Y\_right Y\_final \= **Softmax**(Y\_combined)

Where W represents the learnable weights, and Y\_front, Y\_left, and Y\_right are the outputs from the frontal, left mirror, and right mirror view models, respectively.

This ensemble-based approach offers several advantages. It allows our model to leverage complementary information from different views, potentially capturing aspects of the rider's behavior or environment that may not be visible from a single perspective. Additionally, the learnable weighting mechanism provides a degree of interpretability, as the final weights can give insights into which views are most informative for the intention prediction task.

**Algorithm 3: Process of Multi-view Model**  
**Input**:   
X\_front: (B, T, F) \- Batch of frontal view features  
X\_left: (B, T, F) \- Batch of left mirror view features  
X\_right: (B, T, F) \- Batch of right mirror view features  
       B: Batch size  
       T: Sequence length  
       F: Feature dimension (512 for VGG16 features)  
**Output**:   
Y: (B, C) \- Predicted class probabilities  
        C: Number of classes (rider intentions)  
**Flow**  
1\. Single-View Processing:  
Y\_front: (B, C) ← **MambaModel**(X\_front)  
Y\_left: (B, C) ← **MambaModel**(X\_left)  
Y\_right: (B, C) ← **MambaModel**(X\_right)

2\. Ensemble Weighting:  
W: (3,) ← **LearneableWeights**()  
     
3\. Weighted Combination:  
Y\_combined: (B, C) ← W\[0\] \* Y\_front \+ W\[1\] \* Y\_left \+ W\[2\] \* Y\_right  
     
4\. Final Classification:  
Y: (B, C) ← **Softmax**(Y\_combined)

**Function MambaModel**(X):  
 for each Mamba block:  
        x ← **MambaBlock**(x)  
        x\_pooled: (B, D)           ←**GlobalAveragePooling**(x)  
        logits: (B, C) ← **LinearLayer**(x\_pooled)  
        return logits

#### **3\. Implementation**

**3.1 Data Preprocessing**

For data preprocessing, we utilize pre-extracted VGG16 features for both single-view and multi-view tasks. These features are normalized using z-score normalization (zero mean, unit variance) on a per-sequence basis. This normalization step is crucial for ensuring that the input features are on a consistent scale, which can help with the stability and efficiency of the training process.

We experimented with additional dimensionality reduction or projection of the input features before feeding them into the Mamba models. However, we found that these additional steps did not yield significant performance improvements. As a result, we decided to use the VGG16 features directly, maintaining the original 512-dimensional representation for each frame.

Given that our input sequences (video clips) can have variable lengths, we implement a batch-wise padding strategy where sequences within a batch are padded to the maximum length in that specific batch. This approach allowed us to efficiently process diverse sequence lengths without introducing unnecessary padding across the entire dataset.

**3.2 Training Procedure**  
Both models were trained using the following configuration:  
● Optimizer: AdamW  
● Learning rate: 0.001  
● Weight decay: 1e-5  
● Batch size: 16  
● Number of epochs: 20  
● Learning rate scheduler: StepLR (step\_size=3, gamma=0.8)  
● Loss function: Cross-Entropy Loss  
We implemented early stopping based on validation accuracy, saving the best-performing model during training.  
● d\_model: 512 (matching input dimension)  
● d\_state: 32  
● d\_conv: 4  
● expand: 8

[image1]: <data:image/png;base64,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>
