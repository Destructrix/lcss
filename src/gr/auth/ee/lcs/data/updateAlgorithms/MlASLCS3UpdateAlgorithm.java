/*
 *	Copyright (C) 2011 by Allamanis Miltiadis
 *
 *	Permission is hereby granted, free of charge, to any person obtaining a copy
 *	of this software and associated documentation files (the "Software"), to deal
 *	in the Software without restriction, including without limitation the rights
 *	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *	copies of the Software, and to permit persons to whom the Software is
 *	furnished to do so, subject to the following conditions:
 *
 *	The above copyright notice and this permission notice shall be included in
 *	all copies or substantial portions of the Software.
 *
 *	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *	THE SOFTWARE.
 */
/**
 * 
 */
package gr.auth.ee.lcs.data.updateAlgorithms;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.classifiers.Classifier;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.Macroclassifier;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.geneticalgorithm.IGeneticAlgorithmStrategy;
import gr.auth.ee.lcs.utilities.SettingsLoader;

import java.io.Serializable;
import java.text.DecimalFormat;

/**
 * An alternative MlASLCS update algorithm.
 * 
 * @author Miltiadis Allamanis
 * 
 */
public class MlASLCS3UpdateAlgorithm extends AbstractUpdateStrategy {

	/**
	 * A data object for the MlASLCS3 update algorithms.
	 * 
	 * @author Miltos Allamanis
	 * 
	 */
	final static class MlASLCSClassifierData implements Serializable {

		/**
		 * 
		 */
		private static final long serialVersionUID = 2584696442026755144L;

		/**
		 * d refers to the paper's d parameter in deletion possibility
		 */
		
		public double d = 0;
		
		/**
		 * The classifier's fitness
		 */
		public double fitness = 1; //.5;

		/**
		 * niche set size estimation.
		 */
		public double ns = 1; //20;

		/**
		 * Match Set Appearances.
		 */
		public double msa = 0;

		/**
		 * true positives.
		 */
		public double tp = 0;
		
		/**
		 * totalFitness = numerosity * fitness
		 */
		
		public double totalFitness = 1;
		
		// public double alternateFitness = 1;
		
		public double k = 0;

	}
	
	
	
	/**
	 * The way to differentiate the choice of the fitness calculation formula.
	 * 
	 * Simple = (acc)^n
	 * 
	 * Complex = F + β(k - F)
	 * 
	 * Sharing = F + β((k*num)/(Σ k*num) - F)
	 * 
	 * */
	public static final int FITNESS_MODE_SIMPLE = 0;
	
	public static final int FITNESS_MODE_COMPLEX = 1;
	
	public static final int FITNESS_MODE_SHARING = 2;

	
	
	public static double ACC_0 = (double) SettingsLoader.getNumericSetting("ASLCS_Acc0", .99);

	public static double a = (double) SettingsLoader.getNumericSetting("ASLCS_Alpha", .1);

	/**
	 * The delta (δ) parameter used in determining the formula of possibility of deletion
	 */
	
	public static double DELTA = (double) SettingsLoader.getNumericSetting("ASLCS_DELTA", .1);
	
	/**
	 * The fitness mode, 0 for simple, 1 for complex. 0 As default.
	 * 
	 */
	public static int FITNESS_MODE = (int) SettingsLoader.getNumericSetting("FITNESS_MODE", 0);

	/**
	 * The learning rate.
	 */
	private final double LEARNING_RATE = SettingsLoader.getNumericSetting("LearningRate", 0.2);
	
	
	/**
	 * The theta_del parameter.
	 */
	public static int THETA_DEL = (int) SettingsLoader.getNumericSetting("ASLCS_THETA_DEL", 20);
	
	
	


	/**
	 * The LCS instance being used.
	 */
	private final AbstractLearningClassifierSystem myLcs;

	/**
	 * Genetic Algorithm.
	 */
	public final IGeneticAlgorithmStrategy ga;

	/**
	 * The fitness threshold for subsumption.
	 */
	private final double subsumptionFitnessThreshold;

	/**
	 * The experience threshold for subsumption.
	 */
	private final int subsumptionExperienceThreshold;

	/**
	 * Number of labels used.
	 */
	private final int numberOfLabels;

	/**
	 * The n dumping factor for acc.
	 */
	private final double n;
	
	/**
	 * The mean fitness of the population. 
	 * Updated after fitness updates of the matchset's classifiers, and cover or ga.
	 */
	private double meanPopulationFitness = 0;
	
	
	/**
	 * The sum of d parameters used in the formula for the deletion mechanism.
	 */
	private double sumOfDParameters = 0;

	/**
	 * Constructor.
	 * 
	 * @param lcs
	 *            the LCS being used.
	 * @param labels
	 *            the number of labels
	 * @param geneticAlgorithm
	 *            the GA used
	 * @param nParameter
	 *            the ASLCS dubbing factor
	 * @param fitnessThreshold
	 *            the subsumption fitness threshold to be used.
	 * @param experienceThreshold
	 *            the subsumption experience threshold to be used
	 */
	public MlASLCS3UpdateAlgorithm(final double nParameter,
									final double fitnessThreshold, 
									final int experienceThreshold,
									IGeneticAlgorithmStrategy geneticAlgorithm, 
									int labels,
									AbstractLearningClassifierSystem lcs) {
		
		this.subsumptionFitnessThreshold = fitnessThreshold;
		this.subsumptionExperienceThreshold = experienceThreshold;
		myLcs = lcs;
		numberOfLabels = labels;
		n = nParameter;
		ga = geneticAlgorithm;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#cover(gr.auth.ee.lcs.classifiers
	 * .ClassifierSet, int)
	 */
	@Override
	public void cover(ClassifierSet population, 
					    int instanceIndex) {
		
		final Classifier coveringClassifier = myLcs
											  .getClassifierTransformBridge()
											  .createRandomCoveringClassifier(myLcs.instances[instanceIndex]);
		
		coveringClassifier.created = ga.getTimestamp();
		
		population.addClassifier(new Macroclassifier(coveringClassifier, 1), false);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#createStateClassifierObject()
	 * */
	@Override				

	public Serializable createStateClassifierObject() {
		return new MlASLCSClassifierData();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#getComparisonValue(gr.auth
	 * .ee.lcs.classifiers.Classifier, int)
	 */
	@Override
	public double getComparisonValue(Classifier aClassifier, int mode) {
		
		final MlASLCSClassifierData data = (MlASLCSClassifierData) aClassifier.getUpdateDataObject();
		
		switch (mode) {
		case COMPARISON_MODE_EXPLORATION:
			return ((aClassifier.experience < 10) ? 0 : data.fitness);
			// clean up please
		case COMPARISON_MODE_DELETION:
						
			/*final ClassifierSet population = aClassifier.getLCS().getRulePopulation();
			final int numOfMacroclassifiers = population.getNumberOfMacroclassifiers();*/
			
			// isos 9a eprepe na bei pano, sto for pou upologizo to fitnessSum,
			// alla stin arxi einai ola 0 kai Pdel-->oo
			/*double sumOfDParameters = 0;
			for (int i = 0; i < numOfMacroclassifiers; i++) {
				MlASLCSClassifierData dataForD = (MlASLCSClassifierData) population.getClassifier(i).getUpdateDataObject();
				sumOfDParameters += dataForD.d;
			}*/
			//System.out.println(sumOfDParameters);
			return (double) (data.d / sumOfDParameters);
			
			/*original:
			 * 
			 * return 1 / (data.fitness * ((aClassifier.experience < THETA_DEL) ? 100.
					: Math.exp(-(Double.isNaN(data.ns) ? 1 : data.ns) + 1)));*/
			
		case COMPARISON_MODE_EXPLOITATION:
			//aClassifier.experience < 10) ? 0  kai edo
			final double exploitationFitness = data.fitness; //Math.pow(((data.tp) / (data.msa)) , n); // auto einai to accuracy oxi to fitness
			return (aClassifier.experience < 10) ? 0 : (Double.isNaN(exploitationFitness) ? 0 : exploitationFitness);
		default:
		}
		return 0;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#getData(gr.auth.ee.lcs.classifiers
	 * .Classifier)
	 */
	@Override
	public String getData(Classifier aClassifier) {
		
		final MlASLCSClassifierData data = ((MlASLCSClassifierData) aClassifier.getUpdateDataObject());
		
        DecimalFormat df = new DecimalFormat("#.####");

		return  /* " internalFitness: " + df.format(data.fitness) 
				+ */" tp: " + df.format(data.tp) 
				+ " msa: " + df.format(data.msa) 
				+ " ns: " + df.format(data.ns)
				/*+ " total fitness: " + df.format(data.totalFitness) 
				+ " alt fitness: " + df.format(data.alternateFitness) */ ;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#performUpdate(gr.auth.ee.lcs
	 * .classifiers.ClassifierSet, gr.auth.ee.lcs.classifiers.ClassifierSet)
	 */
	@Override
	public void performUpdate(ClassifierSet matchSet, ClassifierSet correctSet) {
		// Nothing here!
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#setComparisonValue(gr.auth
	 * .ee.lcs.classifiers.Classifier, int, double)
	 */
	@Override
	public void setComparisonValue(Classifier aClassifier, 
									int mode,
									double comparisonValue) {
		
		final MlASLCSClassifierData data = ((MlASLCSClassifierData) aClassifier.getUpdateDataObject());
		data.fitness = comparisonValue;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see gr.auth.ee.lcs.data.AbstractUpdateStrategy#updateSet(gr.auth.ee.lcs.
	 * classifiers.ClassifierSet, gr.auth.ee.lcs.classifiers.ClassifierSet, int,
	 * boolean)
	 */
	@Override
	public void updateSet(ClassifierSet population, 
						   ClassifierSet matchSet,
						   int instanceIndex, 
						   boolean evolve) {

		// Create all label correct sets
		final ClassifierSet[] labelCorrectSets = new ClassifierSet[numberOfLabels];

		for (int i = 0; i < numberOfLabels; i++) { // gia ka9e label parago to correctSet pou antistoixei se auto
			
			labelCorrectSets[i] = generateLabelCorrectSet(matchSet, instanceIndex, i); // periexei tous kanones pou apofasizoun gia to label 9etika.
		}																			   // den perilambanei autous pou adiaforoun.
	
		
		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();
		double sumOfKParameters = 0;

		
		
		// For each classifier in the matchset
		for (int i = 0; i < matchSetSize; i++) { // gia ka9e macroclassifier
			
			final Macroclassifier cl = matchSet.getMacroclassifier(i); // getMacroclassifier => fernei to copy, oxi ton idio ton macroclassifier
			
			int minCurrentNs = Integer.MAX_VALUE;
			final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier.getUpdateDataObject();
			
			int leniency = 0;
			for (int l = 0; l < numberOfLabels; l++) {
				// Get classification ability for label l. an anikei dld sto labelCorrectSet me alla logia.
				final float classificationAbility = cl.myClassifier.classifyLabelCorrectly(instanceIndex, l);

				if (classificationAbility == 0) // an proekupse apo adiaforia
					data.tp += 0.9;
				else if (classificationAbility > 0) { // an proekupse apo 9etiki apofasi (yper)
					data.tp += 1;
					leniency++;
					final int labelNs = labelCorrectSets[l].getTotalNumerosity();
					if (minCurrentNs > labelNs) { // bainei edo mono otan exei prokupsei apo 9etiki apofasi
						minCurrentNs = labelNs;
					}
				}
				data.msa += 1;

			} // kleinei to for gia ka9e label

			
			cl.myClassifier.experience++;
			

			
			/* einai emmesos tropos na elegkso oti o kanonas anikei sto (se ena toulaxiston ennoo) labelCorrectSet
			 * giati to minCurrentNs allazei mono an classificationAbility > 0 dld o kanonas apofasizei, den adiaforei
			 * 
			 * */
			if (minCurrentNs != Integer.MAX_VALUE) {
				//data.ns += .1 * (minCurrentNs - data.ns);
				data.ns += LEARNING_RATE * (minCurrentNs - data.ns);
				
				// an efarmozoume fitness sharing upologise tin parametro k. 
				// o kanonas prepei na exei summetexei se ena klasma apo correctsets
				// gia pano apo ena dinei kala apotelesmata
				if (FITNESS_MODE == FITNESS_MODE_SHARING/* && leniency >= 1/7 * numberOfLabels*/) {
					data.k = Math.pow((data.tp) / (data.msa), n) > ACC_0 ? 1 : a * Math.pow((((data.tp) / (data.msa)) / ACC_0), n);
				}
			}
			
			if (FITNESS_MODE == FITNESS_MODE_SHARING) sumOfKParameters += data.k;
			
			
			switch (FITNESS_MODE) {
			
			case FITNESS_MODE_SIMPLE:
				data.fitness = cl.numerosity * Math.pow((data.tp) / (data.msa), n);
				updateSubsumption(cl.myClassifier);
				break;
			case FITNESS_MODE_COMPLEX:
				data.fitness += LEARNING_RATE * (cl.numerosity * Math.pow((data.tp) / (data.msa), n) - data.fitness);
				updateSubsumption(cl.myClassifier);
				//data.fitness /= cl.numerosity;
			}
		} // kleinei to for gia ka9e macroclassifier
		
		
		
		
		if (FITNESS_MODE == FITNESS_MODE_SHARING) {
			for (int i = 0; i < matchSetSize; i++) {
			
				final Macroclassifier cl = matchSet.getMacroclassifier(i);
				final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier.getUpdateDataObject();
				data.fitness += LEARNING_RATE * (data.k * cl.numerosity / sumOfKParameters - data.fitness);
				updateSubsumption(cl.myClassifier);
				//data.fitness /= cl.numerosity;
			}
		} // kleinei o upologismos tou fitness
		

		final int numOfMacroclassifiers = population.getNumberOfMacroclassifiers();
		

		// calculate the mean fitness of the population, used in the deletion mechanism
		double fitnessSum = 0;
		for (int j = 0; j < numOfMacroclassifiers; j++) {
			fitnessSum += /*population.getClassifierNumerosity(j)
					**/ population.getClassifier(j).getComparisonValue(COMPARISON_MODE_EXPLORATION);
		}
		
		this.meanPopulationFitness = (fitnessSum / numOfMacroclassifiers); // (population.getTotalNumerosity()) ?
		
		
		
		/* update the d parameter, employed in the deletion mechanism, for each classifier in the match set, due to the change in 
		 * the classifiers's numerosities, niches' sizes, fitnesses and the mean fitness of the population
		 */
		 
		for (int i = 0; i < matchSetSize; i++) {
			final Macroclassifier cl = matchSet.getMacroclassifier(i);
			final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier.getUpdateDataObject();

			data.d = data.ns * cl.numerosity * ((cl.myClassifier.experience > THETA_DEL) 
				&& (data.fitness < cl.numerosity * DELTA * meanPopulationFitness) ? 
						meanPopulationFitness / data.fitness : 1);
		}
		
		
		sumOfDParameters = 0;
		for (int j = 0; j < numOfMacroclassifiers; j++) {
			final Macroclassifier cl = population.getMacroclassifier(j);
			final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier.getUpdateDataObject();
			this.sumOfDParameters += data.d;
		}
		
		

		if (evolve) {
			for (int l = 0; l < numberOfLabels; l++) {
				if (labelCorrectSets[l].getNumberOfMacroclassifiers() > 0) {
					ga.evolveSet(labelCorrectSets[l], population);
					population.totalGAInvocations = ga.getTimestamp();
				} else {
					this.cover(population, instanceIndex);
				}
			}
		}
		
				


	}

	/**
	 * Generates the correct set.
	 * 
	 * @param matchSet
	 *            the match set
	 * @param instanceIndex
	 *            the global instance index
	 * @param labelIndex
	 *            the label index
	 * @return the correct set
	 */
	private ClassifierSet generateLabelCorrectSet(final ClassifierSet matchSet,
												   final int instanceIndex, 
												   final int labelIndex) {
		
		final ClassifierSet correctSet = new ClassifierSet(null);
		
		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();
		
		for (int i = 0; i < matchSetSize; i++) {
			
			final Macroclassifier cl = matchSet.getMacroclassifier(i);
			if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) > 0)
				correctSet.addClassifier(cl, false);
		}
		return correctSet;
	}

	/**
	 * Implementation of the subsumption strength.
	 * 
	 * @param aClassifier
	 *            the classifier, whose subsumption ability is to be updated
	 */
	protected void updateSubsumption(final Classifier aClassifier) {
		aClassifier
				.setSubsumptionAbility((aClassifier
						.getComparisonValue(COMPARISON_MODE_EXPLOITATION) / aClassifier.getLCS().getRulePopulation().getClassifierNumerosity(aClassifier) > subsumptionFitnessThreshold)
						&& (aClassifier.experience > subsumptionExperienceThreshold));
	}

}
