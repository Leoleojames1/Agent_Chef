# 🍲Agent Chef (AC) V0.1.1🥘

<img
src="docs/Agent_Chef_logo.png"
  style="display: inline-block; margin: 0 auto; max-width: 50px">

<img
src="docs/agent_chef_poster.jpeg"
  style="display: inline-block; margin: 0 auto; max-width: 50px">

<p align="center">
  <a href="https://ko-fi.com/theborch"><img src="docs/icons/buy me a coffee button.png" height="48"></a>
  <a href="https://discord.gg/dAzSYcnpdF"><img src="docs/icons/Discord button.png" height="48"></a>
</p>

🍲Agent Chef is a powerful tool designed for dataset refinement, structuring, and generation. It empowers users to create high-quality, domain-specific datasets for fine-tuning AI models.🥘

## Features

- 🥕**Dataset Refinement**🥩:
  - Clean and refine your existing datasets
- 🥣**Synthetic Data Generation**🥣:
  - Create procedural and synthetic datasets
- 🔪**Data Poisoning Elimination**🔪:
  - Identify and remove low-quality or malicious data
- 🍛**Specialized Dataset Construction**🍛:
  Generate datasets for specific use cases, including:
  - Function-calling
  - Programming: Python, React, C++
  - Mathematics: LaTeX, Python
  - Languages, Physics, Biology, Chemistry, Law, Cooking, wikipedias, history, context, and more!

## Setup
Start by selecting your ollama model, system prompt, number of duplicates, and the parquet column template format.
<img
src="docs/icons/agent_chef_ui_4.png"
  style="display: inline-block; margin: 0 auto; max-width: 50px">

## Procedural construction
After setup, manually construct your dataset group formatting by wrapping each data point in $("data") wrappers.

<img
src="docs/icons/OARC_commander.png"
  style="display: inline-block; margin: 0 auto; max-width: 50px">
  
These group tags allow agent chef to splice the data out into the correct parquet cells for dataset construction. The automatic formatting tool is currently experimental however it attempts to generate the $("data") tags from the template formatting and inference. Only use this feature if you are daring. 
  [Insert Image]
  
## Synthetic Generation
After procedural dataset construction is complete, you can move on to generating the synthetic dataset from the procedural parquet.
  [Insert Image]
  
## Why Agent Chef?

Agent Chef aims to revolutionize home-brewed AI by providing tools and frameworks that enable users to create high-quality, domain-specific datasets. Whether you're looking to improve an existing dataset or generate new data from scratch, Agent Chef has you covered.

## Getting Started

[Instructions on how to install and set up Agent Chef]

## Usage

[Basic usage instructions and examples]

Agent Chef: Cooking up better datasets for a smarter AI future!
