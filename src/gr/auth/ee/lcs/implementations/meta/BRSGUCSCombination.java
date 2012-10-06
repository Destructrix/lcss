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
package gr.auth.ee.lcs.implementations.meta;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.FoldEvaluator;
import gr.auth.ee.lcs.calibration.InternalValidation;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.populationcontrol.FixedSizeSetWorstFitnessDeletion;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.data.ILCSMetric;
import gr.auth.ee.lcs.data.representations.complex.GenericMultiLabelRepresentation;
import gr.auth.ee.lcs.data.representations.complex.GenericMultiLabelRepresentation.VotingClassificationStrategy;
import gr.auth.ee.lcs.data.updateAlgorithms.SequentialMlUpdateAlgorithm;
import gr.auth.ee.lcs.data.updateAlgorithms.UCSUpdateAlgorithm;
import gr.auth.ee.lcs.evaluators.AccuracyRecallEvaluator;
import gr.auth.ee.lcs.evaluators.ExactMatchEvalutor;
import gr.auth.ee.lcs.evaluators.HammingLossEvaluator;
import gr.auth.ee.lcs.geneticalgorithm.algorithms.SteadyStateGeneticAlgorithm;
import gr.auth.ee.lcs.geneticalgorithm.operators.SinglePointCrossover;
import gr.auth.ee.lcs.geneticalgorithm.operators.UniformBitMutation;
import gr.auth.ee.lcs.geneticalgorithm.selectors.RouletteWheelSelector;
import gr.auth.ee.lcs.utilities.BinaryRelevanceSelector;
import gr.auth.ee.lcs.utilities.ILabelSelector;
import gr.auth.ee.lcs.utilities.LabelFrequencyCalculator;
import gr.auth.ee.lcs.utilities.SettingsLoader;

import java.io.IOException;
import java.util.Arrays;
import java.util.TreeMap;
import java.util.logging.FileHandler;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.Logger;

import weka.core.Instances;

public class BRSGUCSCombination extends AbstractLearningClassifierSystem {

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {

		SettingsLoader.loadSettings();
		final Handler fileLogging = new FileHandler("output.log");

		Logger.getLogger("").setLevel(Level.CONFIG);
		Logger.getLogger("").addHandler(fileLogging);
		final String file = SettingsLoader.getStringSetting("filename", "");

		final BRSGUCSCombination trucs = new BRSGUCSCombination();
		final FoldEvaluator loader = new FoldEvaluator(10, trucs, file);
		loader.evaluate();

	}

	/**
	 * The input file used (.arff).
	 * @uml.property  name="inputFile"
	 */
	private final String inputFile;

	/**
	 * The number of full iterations to train the UCS.
	 * @uml.property  name="iterations"
	 */
	private final int iterations;

	/**
	 * The size of the population to use.
	 * @uml.property  name="populationSize"
	 */
	private final int populationSize;

	/**
	 * The GA crossover rate.
	 * @uml.property  name="cROSSOVER_RATE"
	 */
	private final float CROSSOVER_RATE = (float) SettingsLoader
			.getNumericSetting("crossoverRate", .8);

	/**
	 * Label Selector to be used.
	 * @uml.property  name="selector"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	private final ILabelSelector selector;

	/**
	 * The target label cardinality.
	 * @uml.property  name="targetLC"
	 */
	private final float targetLC;

	/**
	 * The GA mutation rate.
	 * @uml.property  name="mUTATION_RATE"
	 */
	private final double MUTATION_RATE = (float) SettingsLoader
			.getNumericSetting("mutationRate", .04);

	/**
	 * The GA activation rate.
	 * @uml.property  name="tHETA_GA_IMBALANCE_MULTIPLIER"
	 */
	private final int THETA_GA_IMBALANCE_MULTIPLIER = (int) SettingsLoader
			.getNumericSetting("thetaGAImbalanceMultiplier", 10);

	/**
	 * The number of bits to use for representing continuous variables.
	 * @uml.property  name="pRECISION_BITS"
	 */
	private final int PRECISION_BITS = (int) SettingsLoader.getNumericSetting(
			"precisionBits", 5);

	/**
	 * The UCS alpha parameter.
	 * @uml.property  name="uCS_ALPHA"
	 */
	private final double UCS_ALPHA = SettingsLoader.getNumericSetting(
			"UCS_Alpha", .1);

	/**
	 * The UCS n power parameter.
	 * @uml.property  name="uCS_N"
	 */
	private final int UCS_N = (int) SettingsLoader.getNumericSetting("UCS_N",
			10);

	/**
	 * The accuracy threshold parameter.
	 * @uml.property  name="uCS_ACC0"
	 */
	private final double UCS_ACC0 = SettingsLoader.getNumericSetting(
			"UCS_Acc0", .99);

	/**
	 * The learning rate (beta) parameter.
	 * @uml.property  name="uCS_LEARNING_RATE"
	 */
	private final double UCS_LEARNING_RATE = SettingsLoader.getNumericSetting(
			"UCS_beta", .1);

	/**
	 * The UCS experience threshold.
	 * @uml.property  name="uCS_EXPERIENCE_THRESHOLD"
	 */
	private final int UCS_EXPERIENCE_THRESHOLD = (int) SettingsLoader
			.getNumericSetting("UCS_Experience_Theshold", 10);

	/**
	 * The attribute generalization rate.
	 * @uml.property  name="aTTRIBUTE_GENERALIZATION_RATE"
	 */
	private final double ATTRIBUTE_GENERALIZATION_RATE = SettingsLoader
			.getNumericSetting("AttributeGeneralizationRate", 0.33);

	/**
	 * The matchset GA run probability.
	 * @uml.property  name="mATCHSET_GA_RUN_PROBABILITY"
	 */
	private final double MATCHSET_GA_RUN_PROBABILITY = SettingsLoader
			.getNumericSetting("GAMatchSetRunProbability", 0.01);

	/**
	 * Percentage of only updates (and no exploration).
	 * @uml.property  name="uPDATE_ONLY_ITERATION_PERCENTAGE"
	 */
	private final double UPDATE_ONLY_ITERATION_PERCENTAGE = SettingsLoader
			.getNumericSetting("UpdateOnlyPercentage", .1);

	/**
	 * The number of labels used at the dmlUCS.
	 * @uml.property  name="numberOfLabels"
	 */
	private final int numberOfLabels;

	/**
	 * The representation used.
	 * @uml.property  name="rep"
	 * @uml.associationEnd  
	 */
	GenericMultiLabelRepresentation rep;

	/**
	 * The classification strategy.
	 * @uml.property  name="vs"
	 * @uml.associationEnd  
	 */
	VotingClassificationStrategy vs;

	/**
	 * The GA to be used.
	 * @uml.property  name="ga"
	 * @uml.associationEnd  
	 */
	SteadyStateGeneticAlgorithm ga;

	/**
	 * The GA activation rate.
	 * @uml.property  name="tHETA_GA"
	 */
	private final int THETA_GA = (int) SettingsLoader.getNumericSetting(
			"thetaGA", 300);

	/**
	 * Constructor.
	 * 
	 * @throws IOException
	 * 
	 */
	public BRSGUCSCombination() {

		inputFile = SettingsLoader.getStringSetting("filename", "");
		numberOfLabels = (int) SettingsLoader.getNumericSetting(
				"numberOfLabels", 1);
		iterations = (int) SettingsLoader.getNumericSetting("trainIterations",
				1000);
		populationSize = (int) SettingsLoader.getNumericSetting(
				"populationSize", 1500);
		targetLC = (float) SettingsLoader.getNumericSetting(
				"datasetLabelCardinality", 1);
		selector = new BinaryRelevanceSelector(numberOfLabels);

	}

	@Override
	public int[] classifyInstance(double[] instance) {
		return getClassifierTransformBridge().classify(
				this.getRulePopulation(), instance);
	}

	@Override
	public AbstractLearningClassifierSystem createNew() {

		return new BRSGUCSCombination();

	}

	@Override
	public String[] getEvaluationNames() {
		final String[] names = { "Accuracy(pcut)", "Recall(pcut)",
				"HammingLoss(pcut)", "ExactMatch(pcut)", "Accuracy(ival)",
				"Recall(ival)", "HammingLoss(ival)", "ExactMatch(ival)",
				"Accuracy(best)", "Recall(best)", "HammingLoss(best)",
				"ExactMatch(best)" };
		return names;
	}

	@Override
	public double[] getEvaluations(Instances testSet) {
		final double[] results = new double[12];
		Arrays.fill(results, 0);

		final AccuracyRecallEvaluator selfAcc = new AccuracyRecallEvaluator(
				instances, false, this, AccuracyRecallEvaluator.TYPE_ACCURACY);

		final VotingClassificationStrategy str = proportionalCutCalibration();

		final AccuracyRecallEvaluator accEval = new AccuracyRecallEvaluator(
				testSet, false, this, AccuracyRecallEvaluator.TYPE_ACCURACY);
		results[0] = accEval.getMetric(this);

		final AccuracyRecallEvaluator recEval = new AccuracyRecallEvaluator(
				testSet, false, this, AccuracyRecallEvaluator.TYPE_RECALL);
		results[1] = recEval.getMetric(this);

		final HammingLossEvaluator hamEval = new HammingLossEvaluator(testSet,
				false, numberOfLabels, this);
		results[2] = hamEval.getMetric(this);

		final ExactMatchEvalutor testEval = new ExactMatchEvalutor(testSet,
				false, this);
		results[3] = testEval.getMetric(this);

		internalValidationCalibration(selfAcc);

		results[4] = accEval.getMetric(this);
		results[5] = recEval.getMetric(this);
		results[6] = hamEval.getMetric(this);
		results[7] = testEval.getMetric(this);

		useBestClassificationMode();

		results[8] = accEval.getMetric(this);
		results[9] = recEval.getMetric(this);
		results[10] = hamEval.getMetric(this);
		results[11] = testEval.getMetric(this);

		return results;
	}

	public void internalValidationCalibration(ILCSMetric selfAcc) {
		final VotingClassificationStrategy str = rep.new VotingClassificationStrategy(
				(float) SettingsLoader.getNumericSetting(
						"datasetLabelCardinality", 1));
		rep.setClassificationStrategy(str);
		final InternalValidation ival = new InternalValidation(this, str,
				selfAcc);
		ival.calibrate(15);
	}

	public VotingClassificationStrategy proportionalCutCalibration() {
		final VotingClassificationStrategy str = rep.new VotingClassificationStrategy(
				(float) SettingsLoader.getNumericSetting(
						"datasetLabelCardinality", 1));
		rep.setClassificationStrategy(str);

		str.proportionalCutCalibration(this.instances, rulePopulation);
		return str;
	}

	/**
	 * Runs the Direct-ML-UCS.
	 * 
	 */
	@Override
	public void train() {

		// Set BR- variables
		ga = new SteadyStateGeneticAlgorithm(new RouletteWheelSelector(
				AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION, true),
				new SinglePointCrossover(this), CROSSOVER_RATE,
				new UniformBitMutation(MUTATION_RATE), 0, this);

		try {
			rep = new GenericMultiLabelRepresentation(inputFile,
					PRECISION_BITS, numberOfLabels,
					GenericMultiLabelRepresentation.EXACT_MATCH, 0,
					ATTRIBUTE_GENERALIZATION_RATE, this);
		} catch (IOException e) {
			e.printStackTrace();
		}
		vs = rep.new VotingClassificationStrategy(targetLC);
		rep.setClassificationStrategy(vs);

		final UCSUpdateAlgorithm ucsStrategy = new UCSUpdateAlgorithm(
				UCS_ALPHA, UCS_N, UCS_ACC0, UCS_LEARNING_RATE,
				UCS_EXPERIENCE_THRESHOLD, MATCHSET_GA_RUN_PROBABILITY, ga,
				THETA_GA, 1, this);

		this.setElements(rep, ucsStrategy);

		rulePopulation = new ClassifierSet(
				new FixedSizeSetWorstFitnessDeletion(this, numberOfLabels
						* populationSize, new RouletteWheelSelector(
						AbstractUpdateStrategy.COMPARISON_MODE_DELETION, true)));

		// Train BR
		do {
			System.out.println("Training Classifier Set");
			rep.activateLabel(selector);
			TreeMap<String, Integer> fr = LabelFrequencyCalculator
					.createCombinationMap(selector.activeIndexes(),
							numberOfLabels, instances);
			final double imbalance = LabelFrequencyCalculator.imbalanceRate(fr);
			ga.setThetaGA((int) (imbalance * THETA_GA_IMBALANCE_MULTIPLIER));
			ClassifierSet brpopulation = new ClassifierSet(
					new FixedSizeSetWorstFitnessDeletion(
							this,
							populationSize,
							new RouletteWheelSelector(
									AbstractUpdateStrategy.COMPARISON_MODE_DELETION,
									true)));
			trainSet(iterations, brpopulation);
			updatePopulation(
					(int) (iterations * UPDATE_ONLY_ITERATION_PERCENTAGE),
					brpopulation);

			rep.reinforceDeactivatedLabels(brpopulation);
			rulePopulation.merge(brpopulation);

		} while (selector.next());
		rep.activateAllLabels();

		useBestClassificationMode();

		final UCSUpdateAlgorithm updateObj = new UCSUpdateAlgorithm(UCS_ALPHA,
				UCS_N, UCS_ACC0, UCS_LEARNING_RATE, UCS_EXPERIENCE_THRESHOLD,
				SettingsLoader.getNumericSetting("GAMatchSetRunProbability",
						0.01), ga, THETA_GA, 1, this);
		final SequentialMlUpdateAlgorithm strategy = new SequentialMlUpdateAlgorithm(
				updateObj, ga, numberOfLabels);

		this.setElements(rep, strategy);

		trainSet(iterations, rulePopulation);
		updatePopulation((int) (iterations * UPDATE_ONLY_ITERATION_PERCENTAGE),
				rulePopulation);

	}

	public void useBestClassificationMode() {
		rep.setClassificationStrategy(rep.new BestFitnessClassificationStrategy());
	}

}
