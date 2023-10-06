# Medical Knowledge Harmonization: A Graph-based, Entity-Selective Approach to Multi-source Diagnoses

## Description

This project is engineered to formulate an integrated knowledge graph by synthesizing diagnostic data from multiple healthcare centers, thereby providing a comprehensive view of an individual's health trajectory, with a particular emphasis on entities related to Genes, Diseases, Chemicals, Species, Variants, and Cell Types (DNA or RNA), which are notably significant in the context of rare and/or chronic diseases. Leveraging Named Entity Recognition (NER), Entity Normalization, and Relationship Extraction (RE) techniques on raw medical texts, individual knowledge graphs are created and subsequently merged into a unified graph. This exhaustive visualization supports healthcare professionals in making well-informed decisions, ensuring that no detail, especially those pivotal to understanding and managing genetic information and rare diseases, is neglected from any diagnostic source.

## Installation

### Prerequisites

- [Python](https://www.python.org/) (>= 3.x)
- [Conda](https://docs.conda.io/en/latest/)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/anbianchi/knowledge_frombio
   cd knowledge_frombio
2. **Create a Conda environment**:
   ```bash
   conda env create -f environment.yml

3. **Activate the Conda environment**:
  ```bash
  conda activate [Your Environment Name]
  ```

### Usage
You can utilize the tool in two primary ways: by processing the dataset used in the experiments or manually inserting and processing diagnostic reports. Below are the detailed steps for both approaches:

#### 1. Processing the Experiment Dataset

To process the dataset utilized in the experiments, use the following command:

```bash
python main.py --dataset "dataset.csv"

Replace "dataset.csv" with your dataset filename. The script processes the dataset and generates knowledge graphs accordingly.
```

#### 2. Manually Inserting and Processing Diagnostic Reports

If you prefer to manually input diagnostic reports, place your report files within the diagnostic_reports folder. Ensure that all reports within the folder are related to the same patient to maintain consistency and accuracy in the generated knowledge graph.

```bash
python main.py --manual

This command instructs the tool to process the reports present within the diagnostic_reports folder.
```

### Dataset Information

The experiments utilize the [MIMIC-IV-Note: Deidentified free-text clinical notes](https://physionet.org/content/mimic-iv-note/2.2/) dataset, a freely accessible critical care database that holds de-identified health-related data associated with over one thousand patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2008 and 2019. 

#### Key Features:
  
- **De-identification**: Adheres to stringent data security and privacy protocols, ensuring that all patient records are thoroughly de-identified, maintaining the privacy and anonymity of the individuals involved.

- **Accessibility**: The dataset is publicly available to researchers across the world, fostering a collaborative and open research environment.

#### Usage in this Project:
In the context of this project, specifically the "discharge.csv" file, in `notes` folder is used to extract and analyze diagnostic texts. The raw text data from patient reports is processed through our system to generate individual and merged knowledge graphs, which then serve to offer a panoramic view of a patient's medical history and interactions.

#### Accessing the Dataset:
To access and use the MIMIC-IV dataset for replicating our experiments or for your research, please follow the steps below:

1. **Requesting Access**: Visit the [MIMIC website](https://physionet.org/content/mimic-iv-note/2.2/) and follow their guidelines for requesting access to the dataset.

2. **Downloading the Data**: Once approved, download the dataset, specifically the "discharge.csv" file found in the `notes` folder. 

3. **Data Processing**: Use the script `generate.py` from our repository to preprocess the data, converting the notes into a format suitable for our system.

For comprehensive details about the dataset and how to use it, kindly refer to the [official documentation](https://mimic.mit.edu/docs/iv/).

> Note: Even though the dataset is publicly available, we strictly adhere to the usage guidelines provided by MIMIC-IV, ensuring ethical use of the data in our research.

### Code Structure
1. **demo_example/**: Folder containing a subset of results.
2. **modules/**: Folder containing the main script and utility functions.
3. **merged_outputs/** and **temp_outputs/**: Folders where the output graphs and results will be saved.
4. **requirements.txt**: File listing all necessary Python packages.
5. **main_script.py**: Main script to run the program.


