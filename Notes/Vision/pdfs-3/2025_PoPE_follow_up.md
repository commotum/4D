Number of distinct tasks evaluated: 12.
- Indirect Indexing diagnostic character indexing task. (Section 4 Results, p.4)
- Symbolic music modeling on JSB. (Section 4 Results, p.5)
- Symbolic music modeling on MAESTRO. (Section 4 Results, p.5)
- Human Reference Genome next-token prediction. (Section 4 Results, p.5)
- OpenWebText language modeling. (Section 4 Results, p.5)
- LAMBADA zero-shot evaluation. (Section 4 Results, p.6)
- BLiMP zero-shot evaluation. (Section 4 Results, p.6)
- Children's Book Test (CBT) zero-shot evaluation. (Section 4 Results, p.6)
- HellaSwag zero-shot evaluation. (Section 4 Results, p.6)
- PIQA zero-shot evaluation. (Section 4 Results, p.6)
- ARC-E zero-shot evaluation. (Section 4 Results, p.6)
- PG-19 test-time length extrapolation (zero-shot perplexity). (Section 4 Results, p.6)

Number of trained model instances required to cover all tasks: 5.
- Indirect Indexing uses its own trained model. (Section 4 Results, p.4)
- JSB and MAESTRO each require separate music models trained on their datasets. (Section 4 Results, p.5)
- HRG uses its own trained model. (Section 4 Results, p.5)
- A single OpenWebText-pretrained model covers OpenWebText LM plus the six downstream tasks and PG-19 evaluation (zero-shot on OpenWebText-pretrained models). (Section 4 Results, p.5-6)

$$
\boxed{
\frac{12\ \text{tasks}}{5\ \text{models}} = 2.4
}
$$
