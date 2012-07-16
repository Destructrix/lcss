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
package gr.auth.ee.lcs.implementations;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.FoldEvaluator;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.populationcontrol.FixedSizeSetWorstFitnessDeletion;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.data.representations.complex.SingleClassRepresentation;
import gr.auth.ee.lcs.data.updateAlgorithms.ASLCSUpdateAlgorithm;
import gr.auth.ee.lcs.evaluators.ExactMatchEvalutor;
import gr.auth.ee.lcs.evaluators.SingleLabelEvaluator;
import gr.auth.ee.lcs.geneticalgorithm.IGeneticAlgorithmStrategy;
import gr.auth.ee.lcs.geneticalgorithm.algorithms.SteadyStateGeneticAlgorithm;
import gr.auth.ee.lcs.geneticalgorithm.operators.SinglePointCrossover;
import gr.auth.ee.lcs.geneticalgorithm.operators.UniformBitMutation;
import gr.auth.ee.lcs.geneticalgorithm.selectors.RouletteWheelSelector;
import gr.auth.ee.lcs.utilities.SettingsLoader;

import java.io.IOException;

import weka.core.Instances;

/**
 * An AS-LCS implementation.
 * 
 * @author Miltiadis Allamanis
 * 
 */
public final class ASLCS extends AbstractLearningClassifierSystem {
	/**
	 * The main for running AS-LCS.
	 * 
	 * @param args
	 * @throws IOException
	 */
	public static void main(final String[] args) throws IOException {
		SettingsLoader.loadSettings();
		// file = to absolute path tou arxeiou .arff
		// edo diabazo mono to .arff. mesa ston constructor tis ascls 
		// ksanakano loadSettings() gia na diabaso sugkekrimenes metablites
		final String file = SettingsLoader.getStringSetting("filename", "");
		
		final ASLCS aslcs = new ASLCS();
		final FoldEvaluator loader = new FoldEvaluator(/*10*/2, aslcs, file); 
		loader.evaluate(); // trexei ti foldrunnable.run() numOfFolds fores
						   // kaleitai gia na mou aksiologisei pos ta pigan oi kanones sto testSet
	}

	/**
	 * The input file used (.arff).
	 * @uml.property  name="inputFile"
	 */
	private final String inputFile;

	/**
	 * The number of full iterations to train the AS-LCS.
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
	private final float CROSSOVER_RATE = (float) SettingsLoader.getNumericSetting("crossoverRate", .8);

	/**
	 * The GA mutation rate.
	 * @uml.property  name="mUTATION_RATE"
	 */
	private final double MUTATION_RATE = (float) SettingsLoader.getNumericSetting("mutationRate", .04);

	/**
	 * The GA activation rate.
	 * @uml.property  name="tHETA_GA"
	 */
	private final int THETA_GA = (int) SettingsLoader.getNumericSetting("thetaGA", 100);

	/**
	 * The number of bits to use for representing continuous variables.
	 * @uml.property  name="pRECISION_BITS"
	 */
	private final int PRECISION_BITS = (int) SettingsLoader.getNumericSetting("precisionBits", 5);

	/**
	 * The UCS n power parameter.
	 * @uml.property  name="aSLCS_N"
	 */
	private final int ASLCS_N = (int) SettingsLoader.getNumericSetting("ASLCS_N", 10);

	/**
	 * The accuracy threshold parameter.
	 * @uml.property  name="aSLCS_ACC0"
	 */
	private final double ASLCS_ACC0 = SettingsLoader.getNumericSetting("ASLCS_Acc0", .99);

	/**
	 * The UCS experience threshold.
	 * @uml.property  name="aSLCS_EXPERIENCE_THRESHOLD"
	 */
	private final int ASLCS_EXPERIENCE_THRESHOLD = (int) SettingsLoader.getNumericSetting("ASLCS_ExperienceTheshold", 10);

	/**
	 * The attribute generalization rate.
	 * @uml.property  name="aTTRIBUTE_GENERALIZATION_RATE"
	 */
	private final double ATTRIBUTE_GENERALIZATION_RATE = SettingsLoader.getNumericSetting("AttributeGeneralizationRate", 0.33);

	/**
	 * The matchset GA run probability.
	 * @uml.property  name="mATCHSET_GA_RUN_PROBABILITY"
	 */
	private final double MATCHSET_GA_RUN_PROBABILITY = SettingsLoader.getNumericSetting("GAMatchSetRunProbability", /*0.01*/0);

	/**
	 * Percentage of only updates (and no exploration).
	 * @uml.property  name="uPDATE_ONLY_ITERATION_PERCENTAGE"
	 */
	private final double UPDATE_ONLY_ITERATION_PERCENTAGE = SettingsLoader.getNumericSetting("UpdateOnlyPercentage", .1);

	/**
	 * The problem representation.
	 * @uml.property  name="rep"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	private final SingleClassRepresentation rep;

	/**
	 * The AS-LCS constructor.
	 * 
	 * @throws IOException
	 */
	public ASLCS() throws IOException {
		
		inputFile = SettingsLoader.getStringSetting("filename", "");
		iterations = (int) SettingsLoader.getNumericSetting("trainIterations", 1000);
		populationSize = (int) SettingsLoader.getNumericSetting("populationSize", 1500);

		final IGeneticAlgorithmStrategy ga = new SteadyStateGeneticAlgorithm(
				new RouletteWheelSelector(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION, true), 
				new SinglePointCrossover(this), 
				CROSSOVER_RATE,
				new UniformBitMutation(MUTATION_RATE), THETA_GA, this);

		rep = new SingleClassRepresentation(inputFile, PRECISION_BITS, ATTRIBUTE_GENERALIZATION_RATE, this);
		
		rep.setClassificationStrategy(rep.new VotingClassificationStrategy());

		final ASLCSUpdateAlgorithm strategy = new ASLCSUpdateAlgorithm(ASLCS_N,
				ASLCS_ACC0, ASLCS_EXPERIENCE_THRESHOLD,
				MATCHSET_GA_RUN_PROBABILITY, ga, this);

		this.setElements(rep, strategy);

		rulePopulation = new ClassifierSet(
				new FixedSizeSetWorstFitnessDeletion(
						this,
						populationSize,
						new RouletteWheelSelector(
								AbstractUpdateStrategy.COMPARISON_MODE_DELETION,
								true)));
	}

	@Override
	public int[] classifyInstance(double[] instance) {
		return getClassifierTransformBridge().classify(
				this.getRulePopulation(), instance);
	}

	@Override
	public AbstractLearningClassifierSystem createNew() {
		try {
			return new ASLCS();
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
	}

	@Override
	public String[] getEvaluationNames() {
		final String[] names = { "Accuracy" };
		return names;
	}

	@Override
	public double[] getEvaluations(Instances testSet) {
		final double[] result = new double[1];
		final ExactMatchEvalutor testEval = new ExactMatchEvalutor(testSet,
				true, this);
		result[0] = testEval.getMetric(this); 
					/*
					 * 
					 * i parapano klisi tuponei kati san:
					 * Training Fold 0
					 *...........tp:0 fp:4 exactMatch:0.0 total instances:4
					 * Training Fold 1
					 *...........tp:2 fp:2 exactMatch:0.5 total instances:4
					 * (Accuracy: 0.25 auto to tuponei i apo pano sunartisi)

					 * me duo folds.
					 * 
					 *  to result[0] exei tin parametro exactMatch. mia gia ka9e fold.
					 * */
		return result;
	}

	/**
	 * Run the AS-LCS.
	 * 
	 * @throws IOException
	 *             if file not found
	 */
	@Override
	public void train() {

		trainSet(iterations, rulePopulation); // pure explore

		updatePopulation((int) (iterations * UPDATE_ONLY_ITERATION_PERCENTAGE),	//pure exploit?
								rulePopulation);

	}
}
