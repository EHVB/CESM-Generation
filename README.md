# CESM-Generation



## Description
This repo contains Python implementation of Generative models used in our graduation prject in collabroation with Astute Imaging. it should serve as a starting point to train/finetune a model to generate
CESM from DM images.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Training/finetuning with Custom Data](#trainingfinetuning-with-custom-data)

 


## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/EHVB/CESM-Generation.git
    ```
2. Navigate to the project directory:
    ```sh
    cd CESM-Generation
    ```

3. *(Optional)* Create a virtual environment:
    ```sh
    python -m venv venv
    ```
4. *(Optional)* Activate the virtual environment:
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```
5. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training/finetuning with Custom Data

To train the model with your custom data, follow these steps:

1. Prepare your dataset:
    - Ensure your dataset is structured appropriately:
      ```
      dataset/
        Input_dir/
          image000.jpg
          ....
      
        Output_dir/
          image000.jpg
          ....

      ```
      Ensure that each input image corresponds to its correct output image when the folders are sorted
2. The file [train.py](train.py) gives an example of how you would train a CycleGan model on an example dataset.
3. Thefile [continue_train.py](continue_train.py) gives an example of fine tuning a model using saved weights.

   
