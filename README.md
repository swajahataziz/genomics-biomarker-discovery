# Using Deep Learning with Genomics open source Genomics Data (TCGA) for Biomarker Discovery in Amazon SageMaker

There is a growing interest for incorporating genomic data into both research and clinical workflows, such as diagnostic decision support and tailoring disease treatment and prevention strategies to individual genomic characteristics of patients. To be able to use genetic characteristics for such purposes, it is imperative to identify biomarkers, genetic features associated with disease risk, diagnostic, prognosis, or response to treatment. This is an example implementation of feature selection / identification approaches that peform well on the task of biomarker identification on genomic data.

Rather than dealing with millions of features, these approaches can identifying a small number of biomarkers that are relevant for a certain task. This has several advantages, including potentially improving ML model accuracy, reducing risks of under-specified or over-fitted models, rendering tractable complex ML approaches that couldn’t run on millions of features, and aiding scientific research as well as tailored patient support by uncovering highly relevant biomarkers.

We leveraged two genomic datasets for this evaluation, TCGA and GTex. The Cancer Genome Atlas (TCGA) includes a large compendium of thousands of tumors with RNA-seq gene expression measurements of over 20,000 genes. The GTEx dataset consists of samples mostly from healthy subjects and has a higher amount of genomic features (34,218 features).

We have also demonstrated how certain features of Amazon SageMaker can be leveraged to implement cutting edge ML feature selection methods that perform well on tackling these challenges. Specifically:

* Using SageMaker Processing to pre-process data 
* Bringing your own custom containers and algorithms for Model Training
* Using SageMaker managed training
* Using Hyperparameter Optimization

