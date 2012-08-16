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
package gr.auth.ee.lcs;
//comment

import gr.auth.ee.lcs.classifiers.Classifier;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.populationcontrol.SortPopulationControl;
import gr.auth.ee.lcs.classifiers.statistics.MeanAttributeSpecificityStatistic;
import gr.auth.ee.lcs.classifiers.statistics.MeanCoverageStatistic;
import gr.auth.ee.lcs.classifiers.statistics.MeanFitnessStatistic;
import gr.auth.ee.lcs.classifiers.statistics.MeanLabelSpecificity;
import gr.auth.ee.lcs.classifiers.statistics.WeightedMeanAttributeSpecificityStatistic;
import gr.auth.ee.lcs.classifiers.statistics.WeightedMeanCoverageStatistic;
import gr.auth.ee.lcs.classifiers.statistics.WeightedMeanLabelSpecificity;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.data.ClassifierTransformBridge;
import gr.auth.ee.lcs.data.ILCSMetric;
import gr.auth.ee.lcs.evaluators.AccuracyRecallEvaluator;
import gr.auth.ee.lcs.evaluators.ExactMatchEvalutor;
import gr.auth.ee.lcs.evaluators.FileLogger;
import gr.auth.ee.lcs.evaluators.HammingLossEvaluator;
import gr.auth.ee.lcs.utilities.ExtendedBitSet;
import gr.auth.ee.lcs.utilities.SettingsLoader;


import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Vector;

import weka.core.Instances;

/**
 * An abstract LCS class to be implemented by all LCSs.
 * 
 * @author Miltiadis Allamanis
 * 
 */
public abstract class AbstractLearningClassifierSystem {
	
	
	public String hookedMetricsFileDirectory;


	/**
	 * The train set.
	 * @uml.property  name="instances" multiplicity="(0 -1)" dimension="2"
	 */
	public double[][] instances;

	/**
	 * The LCS instance transform bridge.
	 * @uml.property  name="transformBridge"
	 * @uml.associationEnd  
	 */
	
	public double labelCardinality = 1;
	
	public int numberOfCoversOccured = 0;

	
	
	private ClassifierTransformBridge transformBridge;

	/**
	 * The Abstract Update Algorithm Strategy of the LCS.
	 * @uml.property  name="updateStrategy"
	 * @uml.associationEnd  
	 */
	private AbstractUpdateStrategy updateStrategy;

	/**
	 * The rule population.
	 * @uml.property  name="rulePopulation"
	 * @uml.associationEnd  
	 */
	protected ClassifierSet rulePopulation;

	/**
	 * A vector of all evaluator hooks.
	 * @uml.property  name="hooks"
	 * @uml.associationEnd  multiplicity="(0 -1)" elementType="gr.auth.ee.lcs.data.ILCSMetric"
	 */
	private final Vector<ILCSMetric> hooks;

	/**
	 * Frequency of the hook callback execution.
	 * @uml.property  name="hookCallbackRate"
	 */
	private int hookCallbackRate;
	
	
	public int repetition;

	/**
	 * Constructor.
	 * 
	 */ 	
	protected AbstractLearningClassifierSystem() {
		try {
			SettingsLoader.loadSettings();
		} catch (IOException e) {
			e.printStackTrace();
		}
		hooks = new Vector<ILCSMetric>();
		hookCallbackRate = (int) SettingsLoader.getNumericSetting("callbackRate", 100);
	}
	
	
	public void absorbDuplicateClassifiers(ClassifierSet rulePopulation, 
											final boolean evolve) {
		//if (evolve) {
			// if subsumption is only made by the parents and not the whole population, merge classifiers to avoid duplicates
			if (!SettingsLoader.getStringSetting("THOROUGHLY_CHECK_WITH_POPULATION", "true").equals("true")) {
		
				for (int j = 0; j < rulePopulation.getNumberOfMacroclassifiers() ; j++) {
				//for (int j = rulePopulation.getNumberOfMacroclassifiers() -1; j >= 0 ; j--) {

					Vector<Integer> indicesOfDuplicates    = new Vector<Integer>();
					Vector<Float> 	fitnessOfDuplicates    = new Vector<Float>();
					Vector<Integer> experienceOfDuplicates = new Vector<Integer>();

					final Classifier aClassifier = rulePopulation.getMacroclassifiersVector().elementAt(j).myClassifier;
					
					for (int i = rulePopulation.getNumberOfMacroclassifiers() - 1; i >= 0 ; i--) {
					//for (int i = 0; i < rulePopulation.getNumberOfMacroclassifiers(); i++) {

						Classifier theClassifier = rulePopulation.getMacroclassifiersVector().elementAt(i).myClassifier;
						
						if (theClassifier.equals(aClassifier)) { 
							indicesOfDuplicates.add(i);
							float theClassifierFitness = (float) (rulePopulation.getMacroclassifiersVector().elementAt(i).numerosity 
									* getUpdateStrategy().getComparisonValue(theClassifier, AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION));
							fitnessOfDuplicates.add(theClassifierFitness);
							experienceOfDuplicates.add(theClassifier.experience);

						}
					} // exo brei ta indexes ton diplon kanonon sto vector myMacroclassifiers
					
					/*an bro enan mono, simainei oti aClassifier == theClassifier, opote den exei noima na ginei afomoiosi
					 * an bro duo i kai perissoterous simainei oti prepei na epilekso poios apo olous 9a afomoiosei olous tous allous.
					 * opoios exei megalutero fitness afomoionei tous upoloipous. an duo exoun to idio fitness, 9a afomoiosei autos me to megalutero experience
					 * */
					if (indicesOfDuplicates.size() >= 2) { 
						
						int indexOfSurvivor = 0;
						float maxFitness = 0;
						int indexOfClassifierWithMaxFitnessUpTillNow = 0;
						for(int k = 0; k < indicesOfDuplicates.size(); k++) {
							if (fitnessOfDuplicates.elementAt(k) > maxFitness) {
								maxFitness = fitnessOfDuplicates.elementAt(k);
								indexOfSurvivor = k;
								indexOfClassifierWithMaxFitnessUpTillNow = k;
							}
							else if (fitnessOfDuplicates.elementAt(k) == maxFitness) {
								if (experienceOfDuplicates.elementAt(k) >= experienceOfDuplicates.elementAt(indexOfClassifierWithMaxFitnessUpTillNow)) {
									indexOfSurvivor = k;
									indexOfClassifierWithMaxFitnessUpTillNow = k;
								}
									
							}
						}
						// exo brei poios 9a einai o epizon classifier. initiate absorbance
						//for (int k = indicesOfDuplicates.size() -1; k >= 0 ; k--) {
						for (int k = 0; k < indicesOfDuplicates.size() ; k++) {

							if (k != indexOfSurvivor) {
								rulePopulation.getMacroclassifiersVector().elementAt(indicesOfDuplicates.elementAt(indexOfSurvivor)).numerosity += 
									rulePopulation.getMacroclassifiersVector().elementAt(indicesOfDuplicates.elementAt(k)).numerosity;
								rulePopulation.getMacroclassifiersVector().elementAt(indicesOfDuplicates.elementAt(indexOfSurvivor)).numberOfSubsumptions++;
								rulePopulation.deleteMacroclassifier(indicesOfDuplicates.elementAt(k));
							}
						}
						
					}	
					
					if (indicesOfDuplicates.size() != 0) {
						indicesOfDuplicates.clear();
						fitnessOfDuplicates.clear();
						experienceOfDuplicates.clear();
					}
					
				}				
			}
		//}
	}
	

	/**
	 * Classify a single instance.
	 * 
	 * @param instance
	 *            the instance to classify
	 * @return the labels the instance is classified in
	 */
	public abstract int[] classifyInstance(double[] instance);
	
	
	/*
	 * diagrapse tous classifiers pou exoun dei ola ta instances alla den exoun summetasxei se kanena matchSet
	 * 
	 * erxetai me orisma population, olo ton plh9usmo
	 * 
	 * */

	private void cleanUpZeroCoverageClassifiers(final ClassifierSet aSet) {

		final int setSize = aSet.getNumberOfMacroclassifiers();

		for (int i = setSize - 1; i >= 0; i--) { // giati anapoda? giati diagrafei apo vector, an to ekane apo tin arxi 9a diegrafe otinanai
			final Classifier aClassifier = aSet.getClassifier(i);
			final boolean zeroCoverage = (aClassifier.getCheckedInstances() >= instances.length)
				// einai dunaton getCheckedInstances() > instances.length ? see classifier.384 [...]
					&& (aClassifier.getCoverage() == 0);
			if (zeroCoverage) {
				// meta tin allagi stin ClassifierSet.geerateMatchset, opou diagrafetai opoios kanonas exei zero coverage, auti i klisi den exei pleon noima
				//aSet.deleteClassifier(i); // de 9a eprepe na diagrapsei olo ton macroclassifier?
				aSet.deleteMacroclassifier(i);
			}
		}

	}

	/**
	 * Creates a new instance of the actual implementation of the LCS.
	 * 
	 * @return a pointer to the new instance.
	 */
	public abstract AbstractLearningClassifierSystem createNew();

	/**
	 * Execute hooks.
	 * 
	 * @param aSet
	 *            the set on which to run the callbacks
	 */
	private void executeCallbacks(final ClassifierSet aSet, 
								    final int repetition) {
		for (int i = 0; i < hooks.size(); i++) {
			hooks.elementAt(i).getMetric(this);
		}
		
		
		final SortPopulationControl srt = new SortPopulationControl(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION);
		srt.controlPopulation(this.rulePopulation);
		
		int numberOfClassifiersCovered = 0;
		int numberClassifiersGaed = 0;
		int numberOfSubsumptions = 0;
		
		for (int i = 0; i < rulePopulation.getNumberOfMacroclassifiers(); i++) {
			if (this.getRulePopulation().getMacroclassifier(i).myClassifier.getClassifierOrigin() == "cover") {
				numberOfClassifiersCovered++;
			}
			else if (this.getRulePopulation().getMacroclassifier(i).myClassifier.getClassifierOrigin() == "ga") {
				numberClassifiersGaed++;
			}
			numberOfSubsumptions += this.getRulePopulation().getMacroclassifier(i).numberOfSubsumptions;

		}
		
		
		try {
			
			// write the dataset that is being used in file dataset.txt 
			File getConfigurationFile = new File(this.hookedMetricsFileDirectory, "defaultLcs.properties");
			
			if (!getConfigurationFile.exists()) {
				FileInputStream in = new FileInputStream("defaultLcs.properties");
				FileOutputStream out = new FileOutputStream(this.hookedMetricsFileDirectory + "/defaultLcs.properties");
				byte[] buf = new byte[1024];
				int len;
				while ((len = in.read(buf)) > 0) {
				   out.write(buf, 0, len);
				}
				in.close();
				out.close();
			}


			
			// record the rule population and its metrics in population.txt
			final FileWriter fstream = new FileWriter(this.hookedMetricsFileDirectory + "/population.txt", true);
			final BufferedWriter buffer = new BufferedWriter(fstream);
			buffer.write(					
					  String.valueOf(this.repetition) + "th repetition:"
					+ System.getProperty("line.separator")
					+ System.getProperty("line.separator")
					+ "Population size: " + rulePopulation.getNumberOfMacroclassifiers()
					+ System.getProperty("line.separator")
					+ "Timestamp: " + rulePopulation.totalGAInvocations
					+ System.getProperty("line.separator")
					+ "Number of classifiers in population covered :" 	+ numberOfClassifiersCovered
					+ System.getProperty("line.separator")
					+ "Number of classifiers in population ga-ed :" 	+ numberClassifiersGaed
					+ System.getProperty("line.separator")
					+ "Covers occured: " + numberOfCoversOccured
					+ System.getProperty("line.separator")
					+ "Number of subsumptions: " + numberOfSubsumptions
					+ System.getProperty("line.separator")
					+ System.getProperty("line.separator")
					+ rulePopulation
					+ System.getProperty("line.separator"));
			buffer.flush();
			buffer.close();
		} 
		catch (Exception e) {
			e.printStackTrace();
		}	
		
		this.numberOfCoversOccured = 0;
		
		
	}

	/**
	 * Return the LCS's classifier transform bridge.
	 * 
	 * @return the lcs's classifier transform bridge
	 */
	public final ClassifierTransformBridge getClassifierTransformBridge() {
		return transformBridge;
	}

	/**
	 * Returns a string array of the names of the evaluation metrics.
	 * 
	 * @return a string array containing the evaluation names.
	 */
	public abstract String[] getEvaluationNames();

	/**
	 * Returns the evaluation metrics for the given test set.
	 * 
	 * @param testSet
	 *            the test set on which to calculate the metrics
	 * @return a double array containing the metrics
	 */
	public abstract double[] getEvaluations(Instances testSet);
	
	
	/**
	 * A method that computes the label cardinality of the trainset.
	 * 
	 * @author alexandros philotheou
	 * 
	 * */
	
	
	public void getLabelCardinality (double[][] Instances) { 
		
		
		final ClassifierTransformBridge bridge = this.getClassifierTransformBridge();
		
		double sumOfLabels = 0;
		
		for (int i = 0; i < Instances.length; i++) {
			
			final int[] classification = bridge.getDataInstanceLabels(Instances[i]); // px: [1,3,4]. to instance katigoriopoieitai(trainset) sta labels 1,3 kai 4
			//System.out.println(Arrays.toString(classification));
			
			sumOfLabels += classification.length;
			
		}
		
		System.out.println("sumOfLabels:" + sumOfLabels);
		System.out.println("Instances.length:" + Instances.length);

		
		if (Instances.length != 0) {
			this.labelCardinality = (double) (sumOfLabels / Instances.length); 
			System.out.println("labelCardinality:" + this.labelCardinality);

		}
		


	}

	/**
	 * Create a new classifier for the specific LCS.
	 * 
	 * @return the new classifier.
	 */
	public final Classifier getNewClassifier() {
		return Classifier.createNewClassifier(this);
	}

	/**
	 * Return a new classifier object for the specific LCS given a chromosome.
	 * 
	 * @param chromosome
	 *            the chromosome to be replicated
	 * @return a new classifier containing information about the LCS
	 */
	public final Classifier getNewClassifier(final ExtendedBitSet chromosome) {
		return Classifier.createNewClassifier(this, chromosome);
	}

	/**
	 * Getter for the rule population.
	 * @return  a ClassifierSet containing the LCSs population
	 * @uml.property  name="rulePopulation"
	 */
	public final ClassifierSet getRulePopulation() {
		return rulePopulation;
	}

	/**
	 * Returns the LCS's update strategy.
	 * @return  the update strategy
	 * @uml.property  name="updateStrategy"
	 */
	public final AbstractUpdateStrategy getUpdateStrategy() {
		return updateStrategy;
	}

	/**
	 * Prints the population classifiers of the LCS.
	 */
	public final void printSet() {
		rulePopulation.print();
	}

	/**
	 * Register an evaluator to be called during training.
	 * 
	 * @param evaluator
	 *            the evaluator to register
	 * @return true if the evaluator has been registered successfully
	 */
	public final boolean registerHook(final ILCSMetric evaluator) {
		return hooks.add(evaluator);
	}

	/**
	 * Registration of hooks to perform periodical inspection using metrics.
	 * 
	 * @param numberOfLabels 
	 *				the dataset's number of labels. 
	 *
	 *@param instances
	 *			the set of instances on which we will evaluate on. (train or test)
	 *
	 * @author alexandros filotheou
	 * 
	 * 
	 * */
	public void registerMultilabelHooks( double[][] instances, int numberOfLabels) {
				
		this.registerHook(new FileLogger("accuracy",
				new AccuracyRecallEvaluator(instances, false, this, AccuracyRecallEvaluator.TYPE_ACCURACY)));
		
		this.registerHook(new FileLogger("recall",
				new AccuracyRecallEvaluator(instances, false, this, AccuracyRecallEvaluator.TYPE_RECALL)));
		
		this.registerHook(new FileLogger("exactMatch", 
				new ExactMatchEvalutor(instances, false, this)));
		
		this.registerHook(new FileLogger("hamming", 
				new HammingLossEvaluator(instances, false, numberOfLabels, this)));
		
		this.registerHook(new FileLogger("meanFitness",
				new MeanFitnessStatistic(AbstractUpdateStrategy.COMPARISON_MODE_EXPLOITATION)));
		
		this.registerHook(new FileLogger("meanCoverage",
				new MeanCoverageStatistic()));
		
		this.registerHook(new FileLogger("weightedMeanCoverage",
				new WeightedMeanCoverageStatistic(AbstractUpdateStrategy.COMPARISON_MODE_EXPLOITATION)));
		
		this.registerHook(new FileLogger("meanAttributeSpecificity",
				new MeanAttributeSpecificityStatistic()));
		
		this.registerHook(new FileLogger("weightedMeanAttributeSpecificity",
				new WeightedMeanAttributeSpecificityStatistic(AbstractUpdateStrategy.COMPARISON_MODE_EXPLOITATION)));
		
		this.registerHook(new FileLogger("meanLabelSpecificity",
				new MeanLabelSpecificity(numberOfLabels)));
		
		this.registerHook(new FileLogger("weightedMeanLabelSpecificity",
				new WeightedMeanLabelSpecificity(numberOfLabels, AbstractUpdateStrategy.COMPARISON_MODE_EXPLOITATION)));
		
		
	}
	
	
	
	/**
	 * Save the rules to the given filename.
	 * 
	 * @param filename
	 */
	public final void saveRules(String filename) {
		ClassifierSet.saveClassifierSet(rulePopulation, filename);
	}

	/**
	 * Constructor.
	 * 
	 * @param bridge
	 *            the classifier transform bridge
	 * @param update
	 *            the update strategy
	 */
	public final void setElements(final ClassifierTransformBridge bridge,
									final AbstractUpdateStrategy update) {
		transformBridge = bridge;
		updateStrategy = update;
	}

	/**
	 * @param rate
	 * @uml.property  name="hookCallbackRate"
	 */
	public void setHookCallbackRate(int rate) {
		hookCallbackRate = rate;
	}

	
	public void setHookedMetricsFileDirectory(String file) {
		hookedMetricsFileDirectory = file;
	}
	
	/**
	 * Sets the LCS's population.
	 * @param population  the new LCS's population
	 * @uml.property  name="rulePopulation"
	 */
	public final void setRulePopulation(ClassifierSet population) {
		rulePopulation = population;
	}

	/**
	 * Run the LCS and train it.
	 */
	public abstract void train();

	/**
	 * Train population with all train instances and perform evolution.
	 * 
	 * @param iterations
	 *            the number of full iterations (one iteration the LCS is
	 *            trained with all instances) to train the LCS
	 * @param population
	 *            the population of the classifiers to train.
	 */
	protected final void trainSet(final int iterations,
								    final ClassifierSet population) {
		
		trainSet(iterations, population, true); // evolve = true
	}

	/**
	 * Train a classifier set with all train instances.
	 * 
	 * @param iterations
	 *            the number of full iterations (one iteration the LCS is
	 *            trained with all instances) to train the LCS
	 * @param population
	 *            the population of the classifiers to train.
	 * @param evolve
	 *            set true to evolve population, false to only update it
	 *            
	 *            
	 *            ekteleitai gia iterations fores me evolve = true
	 *            kai (int) 0.1 * iterations fores me evolve = false
	 */
	public final void trainSet(final int iterations,
							     final ClassifierSet population, 
							     final boolean evolve) {

		final int numInstances = instances.length;

		repetition = 0;
		
		int trainsBeforeHook = 0;
		//final double instanceProb = (1. / (numInstances));
		while (repetition < iterations) { 		
			System.out.print("[");
			// train  me olo to trainset gia {iterations} fores
			while ((trainsBeforeHook < hookCallbackRate) && (repetition < iterations)) { // ap! allios 9a ksefeuge kai anti na ekteleito gia iterations
				System.out.print('/');													  // 9a ekteleito gia iterations * hookCallBackRate
				for (int i = 0; i < numInstances; i++) {
					trainWithInstance(population, i, evolve); // i pio kato sunartisi
					/*if (Math.random() < instanceProb) // 1/numInstances. ka9e pote prepei na ginetai? skepsou
						cleanUpZeroCoverageClassifiers(population);*/
				}
				repetition++;
				trainsBeforeHook++;
				absorbDuplicateClassifiers(rulePopulation, evolve);
			}
			// check for duplicities every {hookCallbackRate}
			//absorbDuplicateClassifiers(rulePopulation, evolve);

			if (hookCallbackRate < iterations) {
				System.out.print("] ");
				System.out.print("(" + repetition + "/" + iterations + ")");	
			}
			executeCallbacks(population, repetition); 
			trainsBeforeHook = 0;
		}
	}

	/**
	 * Train with instance main template. Trains the classifier set with a
	 * single instance.
	 * 
	 * @param population
	 *            the classifier's population. olos o plh9usmos dld, [P]
	 * @param dataInstanceIndex
	 *            the index of the training data instance
	 * @param evolve
	 *            whether to evolve the set or just train by updating it
	 */
	public final void trainWithInstance(final ClassifierSet population, final int dataInstanceIndex, final boolean evolve) {
		
		final ClassifierSet matchSet = population.generateMatchSet(dataInstanceIndex);

		getUpdateStrategy().updateSet(population, matchSet, dataInstanceIndex, evolve);
		
	}

	/**
	 * Unregister an evaluator.
	 * 
	 * @param evaluator
	 *            the evaluator to register
	 * @return true if the evaluator has been unregisterd successfully
	 */
	public final boolean unregisterEvaluator(final ILCSMetric evaluator) {
		return hooks.remove(evaluator);
	}

	/**
	 * Update population with all train instances but do not perform evolution.
	 * 
	 * @param iterations
	 *            the number of full iterations (one iteration the LCS is
	 *            trained with all instances) to update the LCS
	 * @param population
	 *            the population of the classifiers to update.
	 */
	public final void updatePopulation(final int iterations,
									   final ClassifierSet population) {
		
		trainSet(iterations, population, false); // evolve = false
	}
	
	
	
	

}
