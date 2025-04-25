# Equator Mojo

## Educational Content for the Mojo Programming Language

<p align="center">
  <a href="https://www.gnu.org/licenses/agpl-3.0">
    <img src="https://img.shields.io/badge/License-AGPL%20v3-blue.svg" alt="License: AGPL v3">
  </a>
  <a href="./LICENSE-QCDA">
    <img src="https://img.shields.io/badge/license-Q--CDA-lightgrey.svg" alt="License: Q-CDA">
  </a>
</p>

<div align="center">

## Authors
<p align="center">
Matthew A. Porter, BSc<sup>1</sup>
[![Buy Me A Coffee](https://img.shields.io/badge/Buy_Me_A_Coffee-Support-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/phaedrusflow)
<sup>1</sup>Qompass AI, Spokane, WA  
<div align="center">

<h4 align="center">Connect With Me</h4>
<div align="center">
  
[![Personal LinkedIn](https://img.shields.io/badge/LinkedIn-Matt--Porter-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/matt-a-porter-103535224/)
[![Startup LinkedIn](https://img.shields.io/badge/LinkedIn-Qompass--AI-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/company/95058568/)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--0302--4812-green?style=flat-square&logo=orcid)](https://orcid.org/0000-0002-0302-4812)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Open--Research-blue?style=flat-square&logo=researchgate)](https://www.researchgate.net/profile/Matt-Porter-7)
</div>
<h3 align="center">Social Media</h3>
<div align="center">
  
[![X/Twitter](https://img.shields.io/badge/Twitter-@PhaedrusFlow-blue?style=flat-square&logo=twitter)](https://twitter.com/PhaedrusFlow)
[![Instagram](https://img.shields.io/badge/Instagram-phaedrusflow-purple?style=flat-square&logo=instagram)](https://www.instagram.com/phaedrusflow)
[![YouTube](https://img.shields.io/badge/YouTube-QompassAI-red?style=flat-square&logo=youtube)](https://www.youtube.com/@qompassai)
</div>
<h3 align="center">Developer Programs</h3>
<div align="center">
 
[![NVIDIA Developer](https://img.shields.io/badge/NVIDIA-Developer_Program-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/)
[![Meta Developer](https://img.shields.io/badge/Meta-Developer_Program-0668E1?style=for-the-badge&logo=meta&logoColor=white)](https://developers.facebook.com/)
[![HackerOne](https://img.shields.io/badge/-HackerOne-%23494649?style=for-the-badge&logo=hackerone&logoColor=white)](https://hackerone.com/phaedrusflow)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-qompass-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/qompass)
[![Epic Games Developer](https://img.shields.io/badge/Epic_Games-Developer_Program-313131?style=for-the-badge&logo=epic-games&logoColor=white)](https://dev.epicgames.com/)
</div>

<details id="FAQ">
  <summary><strong>Frequently Asked Questions</strong></summary>

### Q: How do you mitigate against bias?

**TLDR - we do math to make AI ethically useful**

### A: We delineate between mathematical bias (MB) - a fundamental parameter in neural network equations - and algorithmic/social bias (ASB). While MB is optimized during model training through backpropagation, ASB requires careful consideration of data sources, model architecture, and deployment strategies. We implement attention mechanisms for improved input processing and use legal open-source data and secure web-search APIs to help mitigate ASB. 

 [AAMC AI Guidelines | One way to align AI against ASB](https://www.aamc.org/about-us/mission-areas/medical-education/principles-ai-use)

 ### AI Math at a glance

## Forward Propagation Algorithm

$$
y = w_1x_1 + w_2x_2 + ... + w_nx_n + b
$$

Where:

- $y$ represents the model output
- $(x_1, x_2, ..., x_n)$ are input features
- $(w_1, w_2, ..., w_n)$ are feature weights
- $b$ is the bias term
### Neural Network Activation

For neural networks, the bias term is incorporated before activation:

$$
z = \sum_{i=1}^{n} w_ix_i + b
$$
$$
a = \sigma(z)
$$

Where:
- $z$ is the weighted sum plus bias
- $a$ is the activation output
- $\sigma$ is the activation function

### Attention Mechanism- aka what makes the Transformer (The "T" in ChatGPT) powerful

* [Attention High level overview video](https://www.youtube.com/watch?v=fjJOgb-E41w)

* [Attention Is All You Need Arxiv Paper](https://arxiv.org/abs/1706.03762)

The Attention mechanism equation is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

Where:
- $Q$ represents the Query matrix
- $K$ represents the Key matrix
- $V$ represents the Value matrix
- $d_k$ is the dimension of the key vectors
- $\text{softmax}(\cdot)$ normalizes scores to sum to 1

### Q: Do I have to buy a Linux computer to use this? I don't have time for that!
### A: No. You can run Linux and/or the tools we share alongside your existing operating system:
    
* Windows users can use Windows Subsystem for Linux [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
* Mac users can use [Homebrew](https://brew.sh/)
* The code-base instructions were developed with both beginners and advanced users in mind.

### Q: Do you have to get a masters in AI?
### A: Not if you don't want to. To get competent enough to get past ChatGPT dependence at least, you just need a computer and a beginning's mindset. Huggingface is a good place to start. 
* [Huggingface](https://docs.google.com/presentation/d/1IkzESdOwdmwvPxIELYJi8--K3EZ98_cL6c5ZcLKSyVg/edit#slide=id.p)

### Q: What makes a "small" AI model?
### A: AI models ~=10 billion(10B) parameters and below. For comparison, OpenAI's GPT4o contains approximately 200B parameters.

</details>

<details id="Zenodo DOI">
  <summary><strong>Digital Object Identifier (DOI) for public transparency and accountability</strong></summary>
Badge
</details>

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxx)
```

---

- See `.zenodo.json` for metadata, authorship, licensing, and related software attribution.

<details id="Support">
  <summary><strong>How You Can Support</strong></summary>
Badge

If you like Qompass AI and the work we do, consider [buying us a coffee](https://www.buymeacoffee.com/phaedrusflow) ☕ or sponsoring ❤️.

</details>
<details id="Dual-License Notice">
  <summary><strong>What a Dual-License Means</strong></summary>
Badge

### Protection for Vulnerable Populations

The dual licensing aims to address the cybersecurity gap that disproportionately affects underserved populations. As highlighted by recent attacks[^1], low-income residents, seniors, and foreign language speakers face higher-than-average risks of being victims of cyberattacks. By offering both open-source and commercial licensing options, we encourage the development of cybersecurity solutions that can reach these vulnerable groups while also enabling sustainable development and support.

### Preventing Malicious Use

The AGPL-3.0 license ensures that any modifications to the software remain open source, preventing bad actors from creating closed-source variants that could be used for exploitation. This is especially crucial given the rising threats to vulnerable communities, including children in educational settings. The attack on Minneapolis Public Schools, which resulted in the leak of 300,000 files and a $1 million ransom demand, highlights the importance of transparency and security[^8].

### Addressing Cybersecurity in Critical Sectors

The commercial license option allows for tailored solutions in critical sectors such as healthcare, which has seen significant impacts from cyberattacks. For example, the recent Change Healthcare attack[^4] affected millions of Americans and caused widespread disruption for hospitals and other providers. In January 2025, CISA[^2] and FDA[^3] jointly warned of critical backdoor vulnerabilities in Contec CMS8000 patient monitors, revealing how medical devices could be compromised for unauthorized remote access and patient data manipulation.

### Supporting Cybersecurity Awareness

The dual licensing model supports initiatives like the Cybersecurity and Infrastructure Security Agency (CISA) efforts to improve cybersecurity awareness[^7] in "target rich" sectors, including K-12 education[^5]. By allowing both open-source and commercial use, we aim to facilitate the development of tools that support these critical awareness and protection efforts.

### Bridging the Digital Divide

The unfortunate reality is that too many individuals and organizations have gone into a frenzy in every facet of our daily lives[^6]. These unfortunate folks identify themselves with their talk of "10X" returns and building towards Artificial General Intelligence aka "AGI" while offering GPT wrappers. Our dual licensing approach aims to acknowledge this deeply concerning predatory paradigm with clear eyes while still operating to bring the best parts of the open-source community with our services and solutions.

### Recent Cybersecurity Attacks

Recent attacks underscore the importance of robust cybersecurity measures:

- The Change Healthcare cyberattack in February 2024 affected millions of Americans and caused significant disruption to healthcare providers.
- The White House and Congress jointly designated October 2024 as Cybersecurity Awareness Month. This designation comes with over 100 actions that align the Federal government and public/private sector partners are taking to help every man, woman, and child to safely navigate the age of AI.

By offering both open source and commercial licensing options, we strive to create a balance that promotes innovation and accessibility. We address the complex cybersecurity challenges faced by vulnerable populations and critical infrastructure sectors as the foundation of our solutions, not an afterthought..

[^1]: [International Counter Ransomware Initiative 2024 Joint Statement](https://www.whitehouse.gov/briefing-room/statements-releases/2024/10/02/international-counter-ransomware-initiative-2024-joint-statement/)

[^2]: [Contec CMS8000 Contains a Backdoor](https://www.cisa.gov/sites/default/files/2025-01/fact-sheet-contec-cms8000-contains-a-backdoor-508c.pdf)

[^3]: [CISA, FDA warn of vulnerabilities in Contec patient monitors](https://www.aha.org/news/headline/2025-01-31-cisa-fda-warn-vulnerabilities-contec-patient-monitors)

[^4]: [The Top 10 Health Data Breaches of the First Half of 2024](https://www.chiefhealthcareexecutive.com/view/the-top-10-health-data-breaches-of-the-first-half-of-2024)

[^5]: [CISA's K-12 Cybersecurity Initiatives](https://www.cisa.gov/K12Cybersecurity)

[^6]: [Federal Trade Commission Operation AI Comply: continuing the crackdown on overpromises and AI-related lies](https://www.ftc.gov/business-guidance/blog/2024/09/operation-ai-comply-continuing-crackdown-overpromises-ai-related-lies)

[^7]: [A Proclamation on Cybersecurity Awareness Month, 2024 ](https://www.whitehouse.gov/briefing-room/presidential-actions/2024/09/30/a-proclamation-on-cybersecurity-awareness-month-2024/)

[^8]: [Minneapolis school district says data breach affected more than 100,000 people](https://therecord.media/minneapolis-schools-say-data-breach-affected-100000/)
</details>

![Repository Views](https://komarev.com/ghpvc/?username=qompassai-Mojo)
