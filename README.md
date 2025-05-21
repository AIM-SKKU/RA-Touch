<!-- # KOALA: Self-Attention Matters in Knowledge Distillation of Latent Diffusion Models for Memory-Efficient and Fast Image Synthesis -->

# RA-Touch

> **[RA-Touch: Retrieval-Augmented Touch Understanding with Enriched Visual Data](https://arxiv.org/abs/2505.14270)**<br>
> [Yoorhim Cho](https://ofzlo.github.io/)<sup>\*</sup>, [Hongyeob Kim](https://redleaf-kim.github.io/)<sup>\*</sup>, [Semin Kim](https://sites.google.com/g.skku.edu/semin-kim), [Youjia Zhang](https://zhangyj66.github.io/), [Yunseok Choi](https://choiyunseok.github.io/), [Sungeun Hong](https://www.csehong.com/)<sup>†</sup> <br>
> \* Denotes equal contribution <br>
> Sungkyunkwan University <br>

<a href="https://aim-skku.github.io/RA-Touch/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a> &ensp;
<a href="[https://arxiv.org/abs/2312.04005](https://arxiv.org/abs/2505.14270)"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:RA-Touch&color=red&logo=arxiv"></a> &ensp;

## Abstract
### TL;DR
> Tactile Perception using Vison-Language data with ImageNet-T
<details><summary>FULL abstract</summary>
Visuo-tactile perception aims to understand an object’s tactile properties, such as texture, softness, and rigidity. However, the field
remains underexplored because collecting tactile data is costly and labor-intensive. We observe that visually distinct objects can exhibit similar surface textures or material properties. For example, a leather sofa and a leather jacket have different appearances but share similar tactile properties. This implies that tactile understanding can be guided by material cues in visual data, even without direct tactile supervision. In this paper, we introduce RA-Touch, a retrieval-augmented framework that improves visuo-tactile perception by leveraging visual data enriched with tactile semantics. We carefully recaption a large-scale visual dataset with tactile-focused descriptions, enabling the model to access tactile semantics typically absent from conventional visual datasets. A key challenge remains in effectively utilizing these tactile-aware external descriptions. RATouch addresses this by retrieving visual-textual representations aligned with tactile inputs and integrating them to focus on relevant textural and material properties. By outperforming prior methods
on the TVL benchmark, our method demonstrates the potential of retrieval-based visual reuse for tactile understanding.
</details>

## To-Do List
- [ ] Training Code
- [ ] Release Dataset ImgeNet-T

## 📖BibTeX
```bibtex
  @article{
    cho2025ratouchretrievalaugmentedtouchunderstanding,
    title   ={RA-Touch: Retrieval-Augmented Touch Understanding with Enriched Visual Data}, 
    author  ={Yoorhim Cho and Hongyeob Kim and Semin Kim and Youjia Zhang and Yunseok Choi and Sungeun Hong},
    journal ={arXiv preprint arXiv:2505.14270},
    year    ={2025}
  }
```
