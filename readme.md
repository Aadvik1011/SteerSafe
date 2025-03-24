# SteerSafe: Driver Drowsiness Detection System

SteerSafe is a driver drowsiness detection system that leverages machine learning techniques to enhance road safety by monitoring and analyzing driver alertness in real-time.

## Features
- ✨ **Drowsiness Monitoring**: Assesses driver alertness during vehicle operation via analysing and computing the facial features of the driver, furthermore classifying it into classes, being a classification kind of problem statement.
- ⚠️ **Alert System**: Provides immediate warnings to drivers upon detecting signs of drowsiness.
- 💻 **User-Friendly Interface**: Displays real-time feedback and system status through an intuitive graphs and accuracy measures.

## Tech Stack
SteerSafe utilizes the following technologies:

- **Programming Language**: Python
- **Libraries**:
  - NumPy: For numerical computations and analysis of our data once the images are transformed to binary data for data manipulation.
  - Pygame: For handling audio alerts.
  - Pytorch: For building and deploying deep learning models such as the ResNet 152 or Version 1.5 used in our model to build upon a residual neural network for accurate image classification.
- **Machine Learning**:
  - **ResNet 152**: ResNet-152 is a 152-layer deep convolutional neural network that utilizes residual learning to address the vanishing gradient problem, allowing deeper networks to be trained effectively. It was introduced in the ResNet (Residual Networks) family by He et al. in 2015.
  - Key Features of ResNet-152
  -   **Residual Learning**: Uses shortcut (skip) connections to enable training of very deep networks without the degradation problem.
  -   **Bottleneck Blocks**: Each residual block consists of three layers (1x1, 3x3, and 1x1 convolutions) instead of two, making it computationally efficient.
  -   **Deeper but Efficient**: Despite having 152 layers, it has fewer parameters compared to a traditional deep network of the same depth due to bottleneck design.
  - **Custom Layers**: Additional attention-based layers, including **Channel Attention Mechanism**, to enhance feature representation.
  - **Data Augmentation**: Techniques such as flipping, rotation, and brightness adjustment to improve model generalization.

## Installation & Setup

### Prerequisites
Ensure the following are installed:
- **Python**: Version 3.6 or higher.
- **pip**: Python package installer.

### Clone the Repository
```bash
git clone https://github.com/Aadvik1011/SteerSafe.git
cd SteerSafe
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Dataset Preparation
1. **Download the MRL Eye Dataset**: ([gts.ai](https://gts.ai/dataset-download/mrl-infrared-eye-images-dataset/?utm_source=chatgpt.com))
2. **Extract the Dataset**: Unzip the downloaded dataset.
3. **Organize the Data**: Ensure the dataset is structured as follows:

   ```
   data/
     train/
       awake/
       sleepy/
     val/
       awake/
       sleepy/
     test/
       awake/
       sleepy/
   ```

4. **Update Configuration**: Modify the dataset path in the configuration file or script as needed.

### Train the Model
```bash
python train_model.py
```
*Note: Training may require significant computational resources. Ensure your system meets the requirements or consider using cloud-based services.*

### Run the Application
```bash
python drowsiness_detection.py
```

## Model Evaluation
To assess the model's performance, the following metrics and visualizations are used:
- **Confusion Matrix**: Displays model accuracy in predicting awake vs. drowsy states.
- **Classification Report**: Provides precision, recall, and F1-score.
- **ROC Curve & AUC Score**: Evaluates the model's ability to distinguish between classes.
- **Precision-Recall Curve**: Measures classification performance under imbalanced conditions.
- **Normalized Confusion Matrix**: Visualizes prediction distribution.

## Usage Guide
1. **Launch the Application**: Run the `drowsiness_detection.py` script.
2. **Allow Camera Access**: Ensure the system has access to the webcam.
3. **Monitor Alerts**: The system will process the video feed and alert if drowsiness is detected.

## Future Enhancements
- 📱 **OpenCV**: For continuous and real time image and video processing.
- 📊 **Advanced Analytics**: Implement predictive analytics to anticipate drowsiness based on patterns.
- 🔗 **Third-Party Integrations**: Connect with vehicle systems for automated safety responses.

## Contributing
Contributions are welcome! To contribute:

1. **Fork the repository** on GitHub.
2. **Create a new feature branch**: (`git checkout -b feature-branch`).
3. **Implement your changes** and commit them with descriptive messages.
4. **Push your changes** to your forked repository.
5. **Submit a pull request** with a detailed description of your modifications.

## License
SteerSafe is open-source and licensed under the **MIT License**. For more details, see the [LICENSE](LICENSE) file.

## Contact & Support
For inquiries, suggestions, or issues, feel free to reach out via:
- 📧 Email: aadvik1011@gmail.com
- 💬 GitHub Issues: [GitHub Issues Page]
- 🌐 LinkedIn: Aadvik Bharadwaj

*Note: This README is based on the information available from the SteerSafe repository and related sources.*

# ABOUT DATSET:

The Dataset has been augmented for use into our deep learning model to increase efficiency, in order to access the augmented dataset, kindly apply for access :
- [https://drive.google.com/drive/folders/1uSEoanqo_pOmMVaSheq7V6lQFv9PCAAY?usp=share_link]

## MRL Infrared Eye Images Dataset for Drowsiness Detection

This dataset is a **forked version** of the original MRL Eye Dataset, containing infrared eye images categorized into **Awake** and **Sleepy** states. It is split into training, validation, and test sets, comprising over 85,000 images captured under various lighting conditions using multiple sensors. This dataset is tailored for tasks such as eye detection, gaze estimation, blink detection, and drowsiness analysis in computer vision.

## Dataset Structure:
- **Train**: Awake (25,770), Sleepy (25,167)
- **Validation**: Awake (8,591), Sleepy (8,389)
- **Test**: Awake (8,591), Sleepy (8,390)

## Directory Tree:
```data/ train/ awake/ sleepy/ val/ awake/ sleepy/ test/ awake/ sleepy/ ```


## Metadata:
- **subject_id**: Unique identifier for each subject (37 subjects)
- **Attributes**: Eye state, gender, glasses, reflections, lighting, and sensor ID

## Original Dataset Overview

The **MRL Eye Dataset** is a large-scale dataset of human eye images designed for computer vision tasks such as eye detection and blink detection. The original dataset includes 84,898 images captured under various conditions.

## Downloads:
- **Forked Dataset**: [Download Forked Dataset](http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip)
- **Original Dataset**: [Download Original Dataset](http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip)
- **Pupil Annotations**: [Download Pupil Annotations](http://mrl.cs.vsb.cz/data/eyedataset/pupil.txt)

## Contact:
For any questions about the dataset, please contact [Radovan Fusek](http://mrl.cs.vsb.cz//people/fusek/).

## Example Images

The dataset includes both open and closed eye images. Below are examples:

![Eye Image Example](http://mrl.cs.vsb.cz/images/eyedataset/eyedataset01.png)

## References

- [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset)
- [Forked Dataset on Kaggle](https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset)

