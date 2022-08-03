from ipywidgets import interact, fixed, FloatSlider, IntSlider, FloatRangeSlider, VBox, HBox, interactive_output, Dropdown, Checkbox
import matplotlib.pyplot as plt

class convergence:

    def __init__(self, pele_analysis):

        self.pele_analysis = pele_analysis
        self.data = self.pele_analysis.data

    def minimumBindingEnergy(self, initial_threshold=5, max_metric_threshold=30, variable='Binding Energy'):

        def _selectBy(pele_data=None, by_ligand=True, by_protein=False, **metrics):

            n_lowest = IntSlider(
                            value=10,
                            min=1,
                            max=100,
                            description='n_lowest',
                            readout=True,
                            readout_format='.0f')

            if by_ligand:
                ligand_ddm = Dropdown(options=sorted(self.pele_analysis.ligands))
                interact(_byEpoch, pele_data=fixed(pele_data), given_ligand=ligand_ddm,
                         given_protein=fixed(None), n_lowest=n_lowest)

            elif by_protein:
                protein_ddm = Dropdown(options=sorted(self.pele_analysis.proteins))
                interact(_byEpoch, pele_data=fixed(pele_data), given_protein=protein_ddm,
                         given_ligand=fixed(None), n_lowest=n_lowest)

        def _byEpoch(pele_data=None, given_ligand=None, given_protein=None, n_lowest=10):

            if given_ligand:
                ligand_data = pele_data[pele_data.index.get_level_values('Ligand') == given_ligand]

                for protein in ligand_data.index.levels[0]:
                    protein_data = ligand_data[ligand_data.index.get_level_values('Protein') == protein]
                    y_values = []
                    y_errors = []
                    x_values = []
                    for epoch in protein_data.index.levels[2]:
                        epoch_data = protein_data[protein_data.index.get_level_values('Epoch') <= epoch]

                        for metric in metrics:
                            epoch_data = epoch_data[epoch_data[metric] <= metrics[metric]]

                        lowest_values = epoch_data.nsmallest(n_lowest, variable)[variable]
                        y_values.append(lowest_values.mean())
                        y_errors.append(lowest_values.std())
                        x_values.append(epoch)
                    plt.errorbar(x_values, y_values, yerr=y_errors, label=protein, lw=0.8)
                plt.legend(title='Protein', loc='center left', bbox_to_anchor=(1, 0.5))

            if given_protein:
                protein_data = pele_data[pele_data.index.get_level_values('Protein') == given_protein]


                for ligand in protein_data.index.levels[1]:
                    ligand_data = protein_data[protein_data.index.get_level_values('Ligand') == ligand]
                    y_values = []
                    y_errors = []
                    x_values = []
                    for epoch in ligand_data.index.levels[2]:
                        epoch_data = ligand_data[ligand_data.index.get_level_values('Epoch') <= epoch]

                        for metric in metrics:
                            epoch_data = epoch_data[epoch_data[metric] <= metrics[metric]]

                        lowest_values = epoch_data.nsmallest(n_lowest, variable)[variable]
                        y_values.append(lowest_values.mean())
                        y_errors.append(lowest_values.std())
                        x_values.append(epoch)

                    plt.errorbar(x_values, y_values, yerr=y_errors, label=ligand, lw=0.8)
                plt.legend(title='Ligand', loc='center left', bbox_to_anchor=(1, 0.5))

            plt.xlabel('Adaptive PELE Epoch')
            plt.ylabel('Binding Energy [kcal/mol]')
            display(plt.show())

        by_ligand = Checkbox(value=True, description='By Ligand')
        by_protein = Checkbox(value=False, description='By Protein')

        # Add checks for the given pele data pandas df
        metrics = [k for k in self.data.keys() if 'metric_' in k]

        metrics_sliders = {}
        for m in metrics:
            m_slider = FloatSlider(
                            value=initial_threshold,
                            min=0,
                            max=max_metric_threshold,
                            step=0.1,
                            description=m+':',
                            disabled=False,
                            continuous_update=False,
                            orientation='horizontal',
                            readout=True,
                            readout_format='.2f',
                        )
            metrics_sliders[m] = m_slider

        metrics = {m:initial_threshold for m in metrics}

        interact(_selectBy, pele_data=fixed(self.data), by_ligand=by_ligand, by_protein=by_protein, **metrics_sliders)
