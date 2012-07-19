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
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.data.ClassifierTransformBridge;
import gr.auth.ee.lcs.data.ILCSMetric;
import gr.auth.ee.lcs.utilities.ExtendedBitSet;
import gr.auth.ee.lcs.utilities.SettingsLoader;
import gr.auth.ee.lcs.utilities.InstancesUtility;


import java.io.IOException;
import java.util.Arrays;
import java.util.Vector;

import weka.core.Instances;

/**
 * An abstract LCS class to be implemented by all LCSs.
 * 
 * @author Miltiadis Allamanis
 * 
 */
public abstract class AbstractLearningClassifierSystem {

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

		for (int i = setSize - 1; i >= 0; i--) { // giati anapoda?
			final Classifier aClassifier = aSet.getClassifier(i);
			final boolean zeroCoverage = (aClassifier.getCheckedInstances() >= instances.length)
				// einai dunaton getCheckedInstances() > instances.length ? see classifier.384 [...]
					&& (aClassifier.getCoverage() == 0);
			if (zeroCoverage)
				//aSet.deleteClassifier(i); // de 9a eprepe na diagrapsei olo ton macroclassifier?
				aSet.deleteMacroclassifier(i);
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
	private void executeCallbacks(final ClassifierSet aSet) {
		for (int i = 0; i < hooks.size(); i++) {
			hooks.elementAt(i).getMetric(this);
		}
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
	 */
	public final void trainSet(final int iterations,
							     final ClassifierSet population, 
							     final boolean evolve) {

		final int numInstances = instances.length;

		int repetition = 0;
		int trainsBeforeHook = 0;
		final double instanceProb = (1. / (numInstances));
		while (repetition < iterations) { // train  me olo to trainset gia {iterations} fores
			while ((trainsBeforeHook < hookCallbackRate)
					&& (repetition < iterations)) { // ap! allios 9a ksefeuge kai anti na ekteleito gia iterations
				System.out.print('.');				// 9a ekteleito gia iterations * hookCallBackRate

				for (int i = 0; i < numInstances; i++) {
					trainWithInstance(population, i, evolve); // i pio kato sunartisi
					if (Math.random() < instanceProb) // 1/numInstances. ka9e pote prepei na ginetai? skepsou
						cleanUpZeroCoverageClassifiers(population);
				}
				repetition++;
				trainsBeforeHook++;

			}
//			System.out.println("HOOKS:" + hooks.size()); == 0
			executeCallbacks(population); // endiamesa briskei metrikes? mallon einai gia na parakolou9isoume tin proodo --> diagrammata
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
