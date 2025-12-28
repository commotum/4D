Number of distinct tasks evaluated: 32
- Tasks 1-20: bAbI question answering (20 synthetic tasks). (2018_UT.txt, Section 3.1; 2018_UT.txt, Appendix D.1)
- Task 21: Subject-verb agreement. (2018_UT.txt, Section 3.2; 2018_UT.txt, Appendix D.2)
- Task 22: LAMBADA language modeling (task evaluated in LM and reading comprehension settings). (2018_UT.txt, Section 3.3; 2018_UT.txt, Appendix D.3)
- Tasks 23-25: Algorithmic tasks: Copy, Reverse, Addition. (2018_UT.txt, Section 3.4)
- Tasks 26-31: LTE tasks: program, control, addition (program evaluation) and copy, double, reverse (memorization). (2018_UT.txt, Section 3.5; 2018_UT.txt, Appendix D.4)
- Task 32: WMT14 English-German machine translation. (2018_UT.txt, Section 3.6)

Number of trained model instances required to cover all tasks: 13
- bAbI can be handled by a single jointly trained model across all 20 tasks ("train joint"). (2018_UT.txt, Section 3.1; 2018_UT.txt, Appendix D.1)
- Subject-verb agreement is trained/evaluated as its own task. (2018_UT.txt, Section 3.2)
- LAMBADA is trained/evaluated as its own task (with two evaluation settings). (2018_UT.txt, Section 3.3; 2018_UT.txt, Appendix D.3)
- Algorithmic tasks are trained for Copy, Reverse, and Addition. (2018_UT.txt, Section 3.4)
- LTE tasks are trained/evaluated across program/control/addition and copy/double/reverse tasks. (2018_UT.txt, Section 3.5; 2018_UT.txt, Appendix D.4)
- Machine translation is trained on WMT14 En-De as its own task. (2018_UT.txt, Section 3.6)

$$
\boxed{
\frac{32\ \text{tasks}}{13\ \text{models}} = 2.46
}
$$
