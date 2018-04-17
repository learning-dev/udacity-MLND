# Machine Learning Engineer Nanodegree
## Capstone Project
Mohammed Ayaz S


## I. Definition

### Project Overview

Optical Character Recognition is used to convert handwritten or printed text into Machine Encoded text. The aim of Optical Character Recognition (OCR) is to predict the handwritten text with maximum accuracy and minimal errors. Optical Recognition has a wide range of application.

Early Optical Character Recognition was mainly focused towards building tools to aid people who are blind. Since, then Optical Character Recognition has evolved to recognize sign boards for translation, navigation and for text-to-speech. OCR now has a wide range of applications from day-to-day applications to scientific applications.

The aim is here to build a OCR using Bidirectionnal Recurrent Neural Network and compare the accuracy with the Conventional Neural Network

### Problem Statement

**Neural Network** -  one group of algorithms used for machine learning that models the data using graphs of Artificial Neurons, those neurons are a mathematical model that “mimics approximately how a neuron in the brain works”.

 **Recurrent neural network** (RNN) is a class of artificial neural network where connections between units form a directed graph along a sequence. Recurrent Neural Networks (RNNs) are popular models that have shown great promise in many NLP tasks, but in this project - it is used to recognize hand-written character text.

![alt text](https://github.com/learning-dev/udacity-MLND/blob/master/capstone/images/RNN.png)

**Representation of RNN** - [source](https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767) 

**Bidirectional recurrent neural networks**(RNN) are just putting two independent RNNs together. The input sequence is fed in normal time order for one network, and in reverse time order for another. The outputs of the two networks are usually concatenated at each time step, though there are other options, e.g. summation.

![alt text](https://github.com/learning-dev/udacity-MLND/blob/master/capstone/images/bd_RRN.png)

**Representation of  Bidirectional RNN** - [source](http://colah.github.io/posts/2015-09-NN-Types-FP/) 

Implement a model that recognizes a character set using Conventional RNN and also build the same model using Bidirectional Recurrent Neural Network then compare the accuracy of both models.

The objective here is to show that Bidirectional Recurrent Neural Network works better than RNN for OCR and there is an acceptable improvement in the accuracy.

### Metrics

The evaluation metric would be straightforward in the OCR. It is the accuracy of the prediction of the image text to the machine-encoded test. This would depend on various factors like size of the image and the quality of the image. There are labeled images and test images where the predicted label is compared with the labeled image for the accuracy.


## II. Analysis

### Data Exploration

**Dataset**

 The modified version of the MIT OCR dataset is downloaded from the ai.stanford.edu site. This file is called letter.data.gz. Further information on this dataset can be found on the site. This dataset contains a subset of the words in rasterized and normalized form. The original MIT dataset also contained the first letter of each word which was capitalized, and the rest were lowercase. This dataset has had the first character of every word removed so there are no words in capital letters. All words are in lowercase. 

 There are roughly 52,000 character images, and these make up roughly 6800 words.One of the field values is the actual letter that is represented by the image. The Next ID points to the ID of the next letter of the word. The next ID is -1 when it indicates the start of a new word. We don't use the next three columns in this dataset, Word ID, Position, and then the Fold. We don't perform any cross-validation here, so this is not required. We do use the remaining 128 fields, which are the pixel values of the 16x8 images. Each array is 1 image of 128 elements, and these elements can be either 0 or 1. These images correspond to the characters which make up words

### Exploratory Visualization

The total number of words in our dataset - (6877, 6877)

**Tensor shape**
- 6877 words
- Each word padded to have 14 characters
- Each character represented by 16x8 image

Padded Data shape
(6877, 14, -1)

### Algorithms and Techniques

There are no specific algorithm used here. The data is download and reshaped for accordingly. The target features are recognized and data is trained. 

In addition this - we are also building a Recurrent Neural Network.  Every layer in our RNN will have 300 neurons. This is a hyperparameter. The change in the number of neurons according to how the result changes. The long memory cell is used so that we get better performance. The GRU cell is used.

We'll then improve the results that we get with a conventional RNN architecture by using a bidirectional RNN. We have an additional layer called the backward RNN to which we also feed in our image inputs. 

the forward RNN, built using the conventional RNN architecture that we've so far, 14 RNN layers of 300 neurons each. The bidirectional RNN though has an additional RNN layer, which is the backward RNN. This once again has 14 RNN layers of 300 neurons each, but it accepts the input in reverse.

 The outputs of the forward and the backward RNN that we receive are then combined together. This simply stacks the outputs together. This concatenated output is then passed on to the shared softmax for prediction. And this insertion of a backward RNN makes our bidirectional RNN architecture.


### Benchmark

There are various present models and parameters to benchmark with. Here, two models (RNNs and Bi-directional RNNs) are benchmarked with each other test.
The accuracy is compared by plotting a graph.


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing

All words lengths should be the same
- Get every word to be the same length as the longest word in our dataset
- Pad the words with empty characters

![alt text](https://github.com/learning-dev/udacity-MLND/blob/master/capstone/images/code_pad-1.png)

[alt text](https://github.com/learning-dev/udacity-MLND/blob/master/capstone/images/code_pad-2.png)
 
                        *The padded words*

**Tensor shape**
- 6877 words
- Each word padded to have 14 characters
- Each character represented by 16x8 image

**Reshape the data so the image is a 1-D array of pixels**

**Tensor shape**
- 6877 words
- Each an array with 14 characters (padded with empty strings as needed)
**One-hot representation**
- Each character has a feature vector of 26 (only lower case characters)
 `padded_target.shape + (26,)`
 **(6877, 14, 26)**

### Implementation




In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_
    

## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
