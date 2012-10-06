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
import gr.auth.ee.lcs.classifiers.IPopulationControlStrategy;
import gr.auth.ee.lcs.classifiers.Macroclassifier;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.geneticalgorithm.IGeneticAlgorithmStrategy;
import gr.auth.ee.lcs.utilities.SettingsLoader;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.Vector;

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
		
		
		// k for fitness sharing
		public double k = 0;
		
		public int minCurrentNs = 0;
		
		
		public String toString(){
			return 	 "d = " + d 
					+ " fitness = " + fitness
					+ " ns = " + ns
					+ " msa= " + msa
					+ "tp = " + tp
					+ " minCurrentNs = " + minCurrentNs;
		} 
						
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
	public static final int FITNESS_MODE_SIMPLE 	= 0;
	public static final int FITNESS_MODE_COMPLEX 	= 1;
	public static final int FITNESS_MODE_SHARING 	= 2;
	
	
	public static final int DELETION_MODE_DEFAULT = 0;
	public static final int DELETION_MODE_POWER = 1;
	public static final int DELETION_MODE_MILTOS = 2;

	
	
	public static double ACC_0 = (double) SettingsLoader.getNumericSetting("ASLCS_Acc0", .99);
	
	public static double a = (double) SettingsLoader.getNumericSetting("ASLCS_Alpha", .1);
	

	

	/**
	 * The deletion mechanism. 0 for (cl.myClassifier.experience > THETA_DEL) && (data.fitness < DELTA * meanPopulationFitness)
	 * 						   1 for (cl.myClassifier.experience > THETA_DEL) && (Math.pow(data.fitness,n) < DELTA * meanPopulationFitness)	
	 * 
	 * 0 as default
	 * */
		
	public final int DELETION_MODE = (int) SettingsLoader.getNumericSetting("DELETION_MODE", 0);

	/**
	 * The delta (δ) parameter used in determining the formula of possibility of deletion
	 */
	
	public static double DELTA = (double) SettingsLoader.getNumericSetting("ASLCS_DELTA", .1);
	
	/**
	 * The fitness mode, 0 for simple, 1 for complex. 0 As default.
	 */
	public final int FITNESS_MODE = (int) SettingsLoader.getNumericSetting("FITNESS_MODE", 0);
	
	
	/**
	 * do classifiers that don't decide clearly for the label, participate in the correct sets?
	 * */
	public final boolean wildCardsParticipateInCorrectSets = SettingsLoader.getStringSetting("wildCardsParticipateInCorrectSets", "false").equals("true");
	
	
	/** 
	if wildCardsParticipateInCorrectSets is true, and balanceCorrectSets is also true, control the population of the correct sets 
	by examining the numerosity of a correct set comprising only with wildcards against that of a correct set without them.
	if [C#only] <= wildCardParticipationRatio * [C!#], the correct set consists of wildcards AND non-wildcard rules 
	*/

	public final boolean balanceCorrectSets = SettingsLoader.getStringSetting("balanceCorrectSets", "false").equals("true");
	
	public final double wildCardParticipationRatio = SettingsLoader.getNumericSetting("wildCardParticipationRatio", 1);
	
	/**
	 * The learning rate.
	 */
	private final double LEARNING_RATE = SettingsLoader.getNumericSetting("LearningRate", 0.2);
	
	
	/**
	 * The theta_del parameter.
	 */
	public static int THETA_DEL = (int) SettingsLoader.getNumericSetting("ASLCS_THETA_DEL", 20);
	
	
	/**
	 * The MLUCS omega parameter.
	 */	
	private final double OMEGA = SettingsLoader.getNumericSetting("ASLCS_OMEGA", 0.9);
	
	/**
	 * The MLUCS phi parameter.
	 */	
	private final double PHI =  SettingsLoader.getNumericSetting("ASLCS_PHI", 1);


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
	
	
	public int numberOfEvolutionsConducted;
	
	public int numberOfDeletionsConducted;
	
	public int numberOfSubsumptionsConducted;
	
	public int numberOfNewClassifiers;
	
	public long evolutionTime;
	
	public long subsumptionTime;
	
	public long deletionTime;
	

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
		
		/*DELETION_MODE = (int) SettingsLoader.getNumericSetting("DELETION_MODE", 0);
		FITNESS_MODE = (int) SettingsLoader.getNumericSetting("FITNESS_MODE", 0);
		wildCardsParticipateInCorrectSets = SettingsLoader.getStringSetting("wildCardsParticipateInCorrectSets", "true").equals("true");*/
		
		System.out.println("Update algorithm states: ");
		System.out.println("fitness mode: " + FITNESS_MODE);
		System.out.println("deletion mode: " + DELETION_MODE);
		System.out.print("# => [C] " + wildCardsParticipateInCorrectSets);
		if (wildCardsParticipateInCorrectSets) 
			System.out.println(", balance [C]: " + balanceCorrectSets + "\n");
		else
			System.out.println("\n");

	}

	
	/**
	 * 
	 * For every classifier, compute its deletion probability.
	 * 
	 * @param aSet
	 * 			the classifierset of which the classifiers' deletion probabilities we will compute
	 * */
	
	public void computeDeletionProbabilities (ClassifierSet aSet) {

		
		final int numOfMacroclassifiers = aSet.getNumberOfMacroclassifiers();
		
		// calculate the mean fitness of the population, used in the deletion mechanism
		double fitnessSum = 0;
		double meanPopulationFitness = 0;
		
		for (int j = 0; j < numOfMacroclassifiers; j++) {
			fitnessSum += aSet.getClassifierNumerosity(j)
					* aSet.getClassifier(j).getComparisonValue(COMPARISON_MODE_EXPLORATION); 
		}

		meanPopulationFitness = (double) (fitnessSum / aSet.getTotalNumerosity());

		
		/* update the d parameter, employed in the deletion mechanism, for each classifier in the match set, {currently population-wise} due to the change in 
		 * the classifiers's numerosities, niches' sizes, fitnesses and the mean fitness of the population
		 */
		for (int i = 0; i < numOfMacroclassifiers; i++) {
			//final Macroclassifier cl = matchSet.getMacroclassifier(i);
			final Macroclassifier cl = aSet.getMacroclassifier(i);
			final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier.getUpdateDataObject();

			
			if (DELETION_MODE == DELETION_MODE_DEFAULT) {
				data.d = data.ns * ((cl.myClassifier.experience > THETA_DEL) && (data.fitness < DELTA * meanPopulationFitness) ? 
						meanPopulationFitness / data.fitness : 1);	
			
			/* mark the formula responsible for deleting this classifier 
			 * (if exp > theta_del and fitness < delta * <f>) ==> formula = 1, else 0. */
			
				cl.myClassifier.formulaForD = ((cl.myClassifier.experience > THETA_DEL) 
						&& (data.fitness < DELTA * meanPopulationFitness)) ? 1 : 0;
			}
			
			else if (DELETION_MODE == DELETION_MODE_POWER) {

				data.d = data.ns * ((cl.myClassifier.experience > THETA_DEL) && (Math.pow(data.fitness,n) < DELTA * meanPopulationFitness) ? 
							meanPopulationFitness / Math.pow(data.fitness,n) : 1);	
				
				/* mark the formula responsible for deleting this classifier 
				 * (if exp > theta_del and fitness ^ n < delta * <f>) ==> formula = 1, else 0. */
				
				cl.myClassifier.formulaForD = ((cl.myClassifier.experience > THETA_DEL) 
						&& (Math.pow(data.fitness,n) < DELTA * meanPopulationFitness)) ? 1 : 0;
			}
			
			else if (DELETION_MODE == DELETION_MODE_MILTOS) {

				// miltos original
/*				data.d = 1 / (data.fitness * ((cl.myClassifier.experience < THETA_DEL) ? 100.
							: Math.exp(-data.ns  + 1)) );*/
				double acc = data.tp / data.msa;
			
				
				if (cl.myClassifier.experience < THETA_DEL) 
					data.d = 0;//1 / (100 * (Double.isNaN(data.fitness) ? 1 : data.fitness)); // protect the new classifiers
				
				else if (acc >= ACC_0 * (1 - DELTA)) 
					data.d = Math.exp(data.ns / data.fitness * DELTA + 1);
				
				else 
					data.d = Math.exp(data.ns / data.fitness + 1);
					//data.d = Math.pow(data.ns * DELTA, data.ns + 1);
			}
		}	
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
		coveringClassifier.created = myLcs.totalRepetition;//ga.getTimestamp();
		coveringClassifier.setClassifierOrigin("cover"); // o classifier proekupse apo cover
		myLcs.numberOfCoversOccured ++ ;
		population.addClassifier(new Macroclassifier(coveringClassifier, 1), false);
	}
	
	
	private Macroclassifier coverNew( int instanceIndex ) {
		
		final Classifier coveringClassifier = myLcs.getClassifierTransformBridge()
		  .createRandomCoveringClassifier(myLcs.instances[instanceIndex]);
		coveringClassifier.created = myLcs.totalRepetition;//ga.getTimestamp();
		coveringClassifier.setClassifierOrigin("cover"); // o classifier proekupse apo cover
		myLcs.numberOfCoversOccured ++ ;
		return new Macroclassifier(coveringClassifier, 1);
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
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#createStateClassifierObjectArray()
	 * */
	@Override	
	public Serializable[] createClassifierObjectArray() {
		
		MlASLCSClassifierData classifierObjectArray[] = new MlASLCSClassifierData[(int) SettingsLoader.getNumericSetting("numberOfLabels", 1)];
		for (int i = 0; i < numberOfLabels; i++) {
			classifierObjectArray[i] = new MlASLCSClassifierData();
		}
		return classifierObjectArray;
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
		final ClassifierSet correctSetOnlyWildcards = new ClassifierSet(null);
		final ClassifierSet correctSetWithoutWildcards = new ClassifierSet(null);

		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();
		
		for (int i = 0; i < matchSetSize; i++) {
			
			final Macroclassifier cl = matchSet.getMacroclassifier(i);
			
			if (wildCardsParticipateInCorrectSets) {
				
				if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) >= 0) // change: (=) means # => [C]
					correctSet.addClassifier(cl, false);
				
				if (balanceCorrectSets) {
					
					if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) == 0) 
						correctSetOnlyWildcards.addClassifier(cl, false);
					
					if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) > 0)
						correctSetWithoutWildcards.addClassifier(cl, false);
				}
			}
			else 
				if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) > 0)
				correctSet.addClassifier(cl, false);

		}
		
		if (wildCardsParticipateInCorrectSets && balanceCorrectSets) {
			int correctSetWithoutWildcardsNumerosity = correctSetWithoutWildcards.getNumberOfMacroclassifiers();
			int correctSetOnlyWildcardsNumerosity = correctSetOnlyWildcards.getNumberOfMacroclassifiers();
	
			if (correctSetOnlyWildcardsNumerosity <= wildCardParticipationRatio * correctSetWithoutWildcardsNumerosity)
				return correctSet;
			else	
				return correctSetWithoutWildcards;
		}
		
		else return correctSet;
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
			
		case COMPARISON_MODE_DELETION:
			return data.d;
		
		case COMPARISON_MODE_EXPLOITATION:
			return /*(aClassifier.experience < 10) ? 0 :*/ (Double.isNaN(data.tp / data.msa) ? 0 : data.tp / data.msa);//(Double.isNaN(data.fitness) ? 0 : data.fitness);			
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
				+ " d: " + df.format(data.d)
				/*+ " total fitness: " + df.format(data.totalFitness) 
				+ " alt fitness: " + df.format(data.alternateFitness) */ ;
	}

	
	public double getNs (Classifier aClassifier) {
		final MlASLCSClassifierData data = (MlASLCSClassifierData) aClassifier.getUpdateDataObject();
		return data.ns;
	}
	
	public double getAccuracy (Classifier aClassifier) {
		final MlASLCSClassifierData data = (MlASLCSClassifierData) aClassifier.getUpdateDataObject();
		return (Double.isNaN(data.tp / data.msa) ? 0.0 : data.tp / data.msa);
	}
	
	
	
	@Override
	public void inheritParentParameters(Classifier parentA, 
										 Classifier parentB,
										 Classifier child) {
		
		final MlASLCSClassifierData childData = ((MlASLCSClassifierData) child
				.getUpdateDataObject());
		final MlASLCSClassifierData parentAData = ((MlASLCSClassifierData) parentA
				.getUpdateDataObject());
		final MlASLCSClassifierData parentBData = ((MlASLCSClassifierData) parentB
				.getUpdateDataObject());
		childData.ns = (parentAData.ns + parentBData.ns) / 2;
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
	
	
	
	
	/**
	 * Share a the fitness among a set.
	 * 
	 * @param matchSet
	 * 			the match set
	 * 
	 * @param labelCorrectSet
	 *           a correct set in which we share fitness
	 *            
	 * @param l
	 * 			 the index of the label for which the labelCorrectSet is formed
	 * 
	 * @param instanceIndex
	 * 			the index of the instance           
	 * 
	 * @author alexandros philotheou
	 * 
	 */
	private void shareFitness(final ClassifierSet matchSet, 
								final ClassifierSet labelCorrectSet,
								final int l,
								int instanceIndex) {
		
		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();

		double relativeAccuracy = 0;
		
		for (int i = 0; i < matchSetSize; i++) { // gia ka9e macroclassifier
			
			final Macroclassifier cl = matchSet.getMacroclassifier(i); 
			final MlASLCSClassifierData dataArray[] = (MlASLCSClassifierData[]) cl.myClassifier.getUpdateDataArray();
			final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier.getUpdateDataObject();

			// Get classification ability for label l. an anikei sto labelCorrectSet me alla logia.
			final float classificationAbility = cl.myClassifier.classifyLabelCorrectly(instanceIndex, l);
			final int labelNs = labelCorrectSet.getTotalNumerosity();
			
			// update true positives, msa and niche set size
			if (classificationAbility == 0) {// an proekupse apo adiaforia

				dataArray[l].tp += OMEGA;
				dataArray[l].msa += PHI;
				
				data.tp += OMEGA;
				data.msa += PHI;
				
				if (wildCardsParticipateInCorrectSets) {
					
					dataArray[l].minCurrentNs = Integer.MAX_VALUE;

					if (dataArray[l].minCurrentNs > labelNs) 
						dataArray[l].minCurrentNs = labelNs;

					if ((dataArray[l].tp / dataArray[l].msa) > ACC_0) {
						dataArray[l].k = 1;
					}
					else {
						dataArray[l].k = a * Math.pow(((dataArray[l].tp / dataArray[l].msa) / ACC_0), n);
						}
				}
				else
					dataArray[l].k = 0;
					
				
			}
			else if (classificationAbility > 0) { // an proekupse apo 9etiki apofasi 
				dataArray[l].minCurrentNs = Integer.MAX_VALUE;

				dataArray[l].tp += 1;
				data.tp += 1;
				
				if (dataArray[l].minCurrentNs > labelNs) 
					dataArray[l].minCurrentNs = labelNs;
				
				if ((dataArray[l].tp / dataArray[l].msa) > ACC_0) {
					dataArray[l].k = 1;
				}
				else {
					dataArray[l].k = a * Math.pow(((dataArray[l].tp / dataArray[l].msa) / ACC_0), n);
				}	
			}
			else dataArray[l].k = 0;
			
			
			// update msa for positive or negative decision (not updated above)
			if (classificationAbility != 0) {
				dataArray[l].msa += 1;
				data.msa += 1;
			}
			
			 relativeAccuracy += cl.numerosity * dataArray[l].k;
		} // kleinei to for gia ka9e macroclassifier
		
		if (relativeAccuracy == 0) relativeAccuracy = 1;

		for (int i = 0; i < matchSetSize; i++) {
			final Macroclassifier cl = matchSet.getMacroclassifier(i); 
			final MlASLCSClassifierData dataArray[] = (MlASLCSClassifierData[]) cl.myClassifier.getUpdateDataArray();
			dataArray[l].fitness += LEARNING_RATE * (cl.numerosity * dataArray[l].k / relativeAccuracy - dataArray[l].fitness);
			
			//dataArray[l].fitness = Math.pow(dataArray[l].tp / dataArray[l].msa, n); //==> GOOD although too compact
		}
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
		
/*		System.out.println("matchset: ");
		System.out.println(matchSet);*/

		for (int i = 0; i < numberOfLabels; i++) { // gia ka9e label parago to correctSet pou antistoixei se auto
			
			labelCorrectSets[i] = generateLabelCorrectSet(matchSet, instanceIndex, i); // periexei tous kanones pou apofasizoun gia to label 9etika.
			
/*			System.out.println("label: " + i);
			System.out.print("instance: ");
			for (int k = 0; k < myLcs.instances[0].length / 2; k++) {
				System.out.print((int)myLcs.instances[instanceIndex][k]);
			}
			System.out.print("=>");
			for (int k = myLcs.instances[0].length / 2; k < myLcs.instances[0].length; k++) {
				System.out.print((int)myLcs.instances[instanceIndex][k]);
			}
			System.out.println(labelCorrectSets[i]);*/
		
		}																			   

		
		int CorrectSetsPopulation = 0;
		for (int i = 0; i < numberOfLabels; i++) {
			CorrectSetsPopulation += labelCorrectSets[i].getNumberOfMacroclassifiers() ;
		}
		myLcs.meanCorrectSetNumerosity = CorrectSetsPopulation / numberOfLabels;

		
		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();

		
		if (FITNESS_MODE == FITNESS_MODE_SIMPLE || FITNESS_MODE == FITNESS_MODE_COMPLEX) {
			// For each classifier in the matchset
			for (int i = 0; i < matchSetSize; i++) { // gia ka9e macroclassifier
				
				final Macroclassifier cl = matchSet.getMacroclassifier(i); // getMacroclassifier => fernei to copy, oxi ton idio ton macroclassifier
				
				int minCurrentNs = Integer.MAX_VALUE;
				final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier.getUpdateDataObject();
	
				for (int l = 0; l < numberOfLabels; l++) {
					// Get classification ability for label l. an anikei sto labelCorrectSet me alla logia.
					final float classificationAbility = cl.myClassifier.classifyLabelCorrectly(instanceIndex, l);
					final int labelNs = labelCorrectSets[l].getTotalNumerosity();

					if (classificationAbility == 0) {// an proekupse apo adiaforia
						data.tp += OMEGA;
						data.msa += PHI;
						
						if (wildCardsParticipateInCorrectSets) {
							if (minCurrentNs > labelNs) { 
								minCurrentNs = labelNs;
							}
						}
					}
					else if (classificationAbility > 0) { // an proekupse apo 9etiki apofasi (yper)
						data.tp += 1;
						
						if (minCurrentNs > labelNs) { // bainei edo mono otan exei prokupsei apo 9etiki apofasi
							minCurrentNs = labelNs;
						}
					}
					if (classificationAbility != 0) data.msa += 1;
				} // kleinei to for gia ka9e label
	
				cl.myClassifier.experience++;
				
	
				
				/* einai emmesos tropos na elegkso oti o kanonas anikei sto (se ena toulaxiston ennoo) labelCorrectSet
				 * giati to minCurrentNs allazei mono an classificationAbility > 0 dld o kanonas apofasizei, den adiaforei
				 */
				if (minCurrentNs != Integer.MAX_VALUE) {
					//data.ns += .1 * (minCurrentNs - data.ns);
					data.ns += LEARNING_RATE * (minCurrentNs - data.ns);
				}
				
				switch (FITNESS_MODE) {
				
				case FITNESS_MODE_SIMPLE:
					data.fitness = Math.pow((data.tp) / (data.msa), n);
					break;
				case FITNESS_MODE_COMPLEX:
					data.fitness += LEARNING_RATE * (Math.pow((data.tp) / (data.msa), n) - data.fitness);
					
/*					  data.fitness += LEARNING_RATE * (cl.numerosity * Math.pow((data.tp) / (data.msa), n) - data.fitness);
					  data.fitness /= cl.numerosity;*/
					 
					break;
				}
				updateSubsumption(cl.myClassifier);

			} // kleinei to for gia ka9e macroclassifier
		}
		
		
		
		else if (FITNESS_MODE == FITNESS_MODE_SHARING) {
			
			for (int l = 0; l < numberOfLabels; l++) {
				shareFitness(matchSet, labelCorrectSets[l], l, instanceIndex);
			} 
			
			for (int i = 0; i < matchSetSize; i++) { 
				final Macroclassifier cl = matchSet.getMacroclassifier(i);	
				cl.myClassifier.experience++; 
				final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier.getUpdateDataObject();
				final MlASLCSClassifierData dataArray[] = (MlASLCSClassifierData[]) cl.myClassifier.getUpdateDataArray();
				
				double fitnessSum = 0;
				double ns = 0;
				
				for (int l = 0; l < numberOfLabels; l++) {
					fitnessSum += dataArray[l].fitness;	
					ns += dataArray[l].minCurrentNs;
				}
				ns /= numberOfLabels;
				data.fitness = (fitnessSum / cl.numerosity) / numberOfLabels;

				if (ns != Integer.MAX_VALUE) {
					//data.ns += .1 * (minCurrentNs - data.ns);
					data.ns += LEARNING_RATE * (ns - data.ns);
				}
					
				if (Math.pow(data.tp / data.msa, n) > ACC_0) {
					if (cl.myClassifier.experience >= this.subsumptionExperienceThreshold)
						cl.myClassifier.setSubsumptionAbility(true);
				}
				else {
					cl.myClassifier.setSubsumptionAbility(false);
				}
				
			} 
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
	
	
	
	
	
	@Override
	public void updateSetNew(ClassifierSet population, 
							   ClassifierSet matchSet,
							   int instanceIndex, 
							   boolean evolve) {
		
		// Create all label correct sets
		final ClassifierSet[] labelCorrectSets = new ClassifierSet[numberOfLabels];
		
/*		System.out.println("matchset: ");
		System.out.println(matchSet);*/

		for (int i = 0; i < numberOfLabels; i++) { // gia ka9e label parago to correctSet pou antistoixei se auto
			
			labelCorrectSets[i] = generateLabelCorrectSet(matchSet, instanceIndex, i); // periexei tous kanones pou apofasizoun gia to label 9etika.
			
/*			System.out.println("label: " + i);
			System.out.print("instance: ");
			for (int k = 0; k < myLcs.instances[0].length / 2; k++) {
				System.out.print((int)myLcs.instances[instanceIndex][k]);
			}
			System.out.print("=>");
			for (int k = myLcs.instances[0].length / 2; k < myLcs.instances[0].length; k++) {
				System.out.print((int)myLcs.instances[instanceIndex][k]);
			}
			System.out.println(labelCorrectSets[i]);*/
		
		}																			   

		
		int CorrectSetsPopulation = 0;
		for (int i = 0; i < numberOfLabels; i++) {
			CorrectSetsPopulation += labelCorrectSets[i].getNumberOfMacroclassifiers() ;
		}
		myLcs.meanCorrectSetNumerosity = CorrectSetsPopulation / numberOfLabels;

		
		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();

		
		if (FITNESS_MODE == FITNESS_MODE_SIMPLE || FITNESS_MODE == FITNESS_MODE_COMPLEX) {
			// For each classifier in the matchset
			for (int i = 0; i < matchSetSize; i++) { // gia ka9e macroclassifier
				
				final Macroclassifier cl = matchSet.getMacroclassifier(i); // getMacroclassifier => fernei to copy, oxi ton idio ton macroclassifier
				
				int minCurrentNs = Integer.MAX_VALUE;
				final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier.getUpdateDataObject();
	
				for (int l = 0; l < numberOfLabels; l++) {
					// Get classification ability for label l. an anikei sto labelCorrectSet me alla logia.
					final float classificationAbility = cl.myClassifier.classifyLabelCorrectly(instanceIndex, l);
					final int labelNs = labelCorrectSets[l].getTotalNumerosity();

					if (classificationAbility == 0) {// an proekupse apo adiaforia
						data.tp += OMEGA;
						data.msa += PHI;
						
						if (wildCardsParticipateInCorrectSets) {
							if (minCurrentNs > labelNs) { 
								minCurrentNs = labelNs;
							}
						}
					}
					else if (classificationAbility > 0) { // an proekupse apo 9etiki apofasi (yper)
						data.tp += 1;
						
						if (minCurrentNs > labelNs) { // bainei edo mono otan exei prokupsei apo 9etiki apofasi
							minCurrentNs = labelNs;
						}
					}
					if (classificationAbility != 0) data.msa += 1;
				} // kleinei to for gia ka9e label
	
				cl.myClassifier.experience++;
				
	
				
				/* einai emmesos tropos na elegkso oti o kanonas anikei sto (se ena toulaxiston ennoo) labelCorrectSet
				 * giati to minCurrentNs allazei mono an classificationAbility > 0 dld o kanonas apofasizei, den adiaforei
				 */
				if (minCurrentNs != Integer.MAX_VALUE) {
					//data.ns += .1 * (minCurrentNs - data.ns);
					data.ns += LEARNING_RATE * (minCurrentNs - data.ns);
				}
				
				switch (FITNESS_MODE) {
				
				case FITNESS_MODE_SIMPLE:
					data.fitness = Math.pow((data.tp) / (data.msa), n);
					break;
				case FITNESS_MODE_COMPLEX:
					data.fitness += LEARNING_RATE * (Math.pow((data.tp) / (data.msa), n) - data.fitness);
					
/*					  data.fitness += LEARNING_RATE * (cl.numerosity * Math.pow((data.tp) / (data.msa), n) - data.fitness);
					  data.fitness /= cl.numerosity;*/
					 
					break;
				}
				updateSubsumption(cl.myClassifier);

			} // kleinei to for gia ka9e macroclassifier
		}
		
		
		
		else if (FITNESS_MODE == FITNESS_MODE_SHARING) {
			
			for (int l = 0; l < numberOfLabels; l++) {
				shareFitness(matchSet, labelCorrectSets[l], l, instanceIndex);
			} 
			
			for (int i = 0; i < matchSetSize; i++) { 
				final Macroclassifier cl = matchSet.getMacroclassifier(i);	
				cl.myClassifier.experience++; 
				final MlASLCSClassifierData data = (MlASLCSClassifierData) cl.myClassifier.getUpdateDataObject();
				final MlASLCSClassifierData dataArray[] = (MlASLCSClassifierData[]) cl.myClassifier.getUpdateDataArray();
				
				double fitnessSum = 0;
				double ns = 0;
				
				for (int l = 0; l < numberOfLabels; l++) {
					fitnessSum += dataArray[l].fitness;	
					ns += dataArray[l].minCurrentNs;
				}
				ns /= numberOfLabels;
				data.fitness = (fitnessSum / cl.numerosity) / numberOfLabels;

				if (ns != Integer.MAX_VALUE) {
					//data.ns += .1 * (minCurrentNs - data.ns);
					data.ns += LEARNING_RATE * (ns - data.ns);
				}
					
				if (Math.pow(data.tp / data.msa, n) > ACC_0) {
					if (cl.myClassifier.experience >= this.subsumptionExperienceThreshold)
						cl.myClassifier.setSubsumptionAbility(true);
				}
				else {
					cl.myClassifier.setSubsumptionAbility(false);
				}
				
			} 
		}	
		
		
		numberOfEvolutionsConducted = 0;
		numberOfSubsumptionsConducted = 0;
		numberOfDeletionsConducted = 0;
		numberOfNewClassifiers = 0;
		evolutionTime = 0;
		subsumptionTime = 0;
		deletionTime = 0;
		
		if (evolve) {
			
			evolutionTime = -System.currentTimeMillis();
						
			Vector<Integer> labelsToEvolve = new Vector<Integer>();
			
			Vector<Integer> labelsToCover = new Vector<Integer>();
			
			for (int l = 0; l < numberOfLabels; l++) {
				if (labelCorrectSets[l].getNumberOfMacroclassifiers() > 0) {
					
					ga.increaseTimestamp();
					int meanAge = ga.getMeanAge(labelCorrectSets[l]);
					if ( !( ga.getTimestamp() - meanAge < ga.getActivationAge()) )
					{
						labelsToEvolve.add(l);
						for ( int i = 0; i < labelCorrectSets[l].getNumberOfMacroclassifiers(); i++ )
						{
							labelCorrectSets[l].getClassifier(i).timestamp = ga.getTimestamp();
						}
					}					
				} else {
					labelsToCover.add(l);
				}
			}
			
			numberOfEvolutionsConducted = labelsToEvolve.size();
			
			Vector<Integer> indicesToSubsume = new Vector<Integer>();
			
			ClassifierSet newClassifiersSet = new ClassifierSet(null);
			
			for( int l = 0; l < numberOfLabels ; l++)
			{
				if(labelsToEvolve.contains(l))
				{
					ga.evolveSetNew(labelCorrectSets[l],population);
					indicesToSubsume.addAll(ga.getIndicesToSubsume());
					newClassifiersSet.merge(ga.getNewClassifiersSet());
					
					//numberOfSubsumptionsConducted += ga.getIndicesToSubsume().size();
					//numberOfNewClassifiers += ga.getNewClassifiersSet().getNumberOfMacroclassifiers();
					
					subsumptionTime += ga.getSubsumptionTime();
					
					labelsToEvolve.removeElement(l);
				}
				
				if(labelsToCover.contains(l))
				{
					newClassifiersSet.addClassifier(this.coverNew(instanceIndex), false);
					labelsToCover.removeElement(l);
				}
			}
			
			population.totalGAInvocations = ga.getTimestamp();

			
			numberOfSubsumptionsConducted = indicesToSubsume.size();
			numberOfNewClassifiers        = newClassifiersSet.getNumberOfMacroclassifiers();
			
			for ( int i = 0; i < indicesToSubsume.size() ; i++ )
			{
				population.getMacroclassifiersVector().elementAt(indicesToSubsume.elementAt(i)).numerosity++; // get vector
				population.totalNumerosity++;
			}
			
			population.mergeWithoutControl(newClassifiersSet);
			
			deletionTime = -System.currentTimeMillis();
			final IPopulationControlStrategy theControlStrategy = population.getPopulationControlStrategy();
			theControlStrategy.controlPopulation(population);
			deletionTime += System.currentTimeMillis();
			
			numberOfDeletionsConducted = theControlStrategy.getNumberOfDeletionsConducted();
			
			evolutionTime += System.currentTimeMillis();
			
		}
		
	}

	




	/**
	 * Implementation of the subsumption strength.
	 * 
	 * @param aClassifier
	 *            the classifier, whose subsumption ability is to be updated
	 */
	protected void updateSubsumption(final Classifier aClassifier) {
		aClassifier.setSubsumptionAbility(
				(aClassifier.getComparisonValue(COMPARISON_MODE_EXPLOITATION) > subsumptionFitnessThreshold)
						&& (aClassifier.experience > subsumptionExperienceThreshold));
	}


}