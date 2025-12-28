1. Number of distinct tasks evaluated: 27.
   - 19 pretraining datasets (Table 5): acoustic scattering discontinuous; acoustic scattering inclusions; acoustic scattering maze; active matter; euler multiquadrants periodicBC; euler multiquadrants openBC; gray scott reaction diffusion; helmholtz staircase; MHD (3D); planetswe; rayleigh benard; rayleigh taylor instability; shear flow; supernova explosion; turbulence gravity cooling; turbulent radiative layer 2D; turbulent radiative layer 3D; viscoelastic instability; FPOHarmonics. (Notes/Vision/pdfs-3/.2025_WALRUS.extracted.txt:1934; Notes/Vision/pdfs-3/.2025_WALRUS.extracted.txt:2093; Notes/Vision/pdfs-3/.2025_WALRUS.extracted.txt:2112)
   - 8 downstream datasets (Table 6): FPOSkelenton; PoolBoil Subcooled; Conditioned Incompressible NS; CE-RM; CNS Turbulent; CNS Random; post neutron star merger; convective envelope rsg. (Notes/Vision/pdfs-3/.2025_WALRUS.extracted.txt:2122; Notes/Vision/pdfs-3/.2025_WALRUS.extracted.txt:2130)

2. Number of trained model instances required to cover all tasks: 27.
   - Cross-domain analysis fine-tunes Walrus separately for each pretraining dataset ("We specialize Walrus to each dataset through an additional 500K samples of finetuning"), so each of the 19 pretraining tasks requires its own trained model instance. (Notes/Vision/pdfs-3/.2025_WALRUS.extracted.txt:947; Notes/Vision/pdfs-3/.2025_WALRUS.extracted.txt:951)
   - Downstream evaluation fine-tunes Walrus on each new dataset exclusively ("For all finetuning tasks, Walrus was trained for an additional 50 pseudo-epochs ... on the new dataset exclusively"), so each of the 8 downstream tasks requires its own trained model instance. (Notes/Vision/pdfs-3/.2025_WALRUS.extracted.txt:1963; Notes/Vision/pdfs-3/.2025_WALRUS.extracted.txt:1964)

3. Task-Model Ratio:
$$
\boxed{
\frac{27\ \text{tasks}}{27\ \text{models}} = 1
}
$$
