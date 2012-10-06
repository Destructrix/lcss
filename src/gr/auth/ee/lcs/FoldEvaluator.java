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

//Test GIT with comment. Fani.
//Test GIT with second comment. Fani.
//Test GIT with third comment. Fani.
//This is the last comment before merging back to master. 

/**
 * 
 */
package gr.auth.ee.lcs;

import gr.auth.ee.lcs.utilities.InstancesUtility;
import gr.auth.ee.lcs.utilities.SettingsLoader;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import weka.core.Instances;

/**
 * n-fold evaluator.
 * 
 * @author Miltiadis Allamanis
 * 
 */
public class FoldEvaluator {

	/**
	 * The number of labels of the dataset 
	 */
	final int numberOfLabels = (int) SettingsLoader.getNumericSetting("numberOfLabels", 1); 
	
	final boolean initializePopulation = SettingsLoader.getStringSetting("initializePopulation", "false").equals("true");
	final String file = SettingsLoader.getStringSetting("filename", ""); // to trainSet.arff


	
	/**
	 * An inner class representing a fold of training. To be run as a thread
	 * 
	 * @author Miltiadis Allamanis
	 * 
	 */
	private final class FoldRunnable implements Runnable {
		private final int metricOptimizationIndex;
		private final int i;
		private final int numOfFoldRepetitions;

		/**
		 * The train set.
		 */
		private Instances trainSet;

		/**
		 * The test set.
		 */
		private Instances testSet;

		/**
		 * Constructor
		 * 
		 * @param metricOptimizationIndex
		 *            the index of the metric to optimize on
		 * @param nFold
		 *            the fold number
		 * @param numOfFoldRepetitions
		 *            the number of repetitions to perform
		 */
		private FoldRunnable(int metricOptimizationIndex, int nFold,
				int numOfFoldRepetitions) {
			this.metricOptimizationIndex = metricOptimizationIndex;
			this.i = nFold;
			this.numOfFoldRepetitions = numOfFoldRepetitions;
		}

		@Override
		public void run() {
			
			// mark commencement time in console
			final Calendar cal = Calendar.getInstance();
			final SimpleDateFormat sdf = new SimpleDateFormat("kk:mm:ss, dd/MM/yyyy");
			String timestampStart = sdf.format(cal.getTime());
			System.out.println("Execution started @ " + timestampStart + "\n");
			
			
			double[][] results = new double[numOfFoldRepetitions][];
						
			for (int repetition = 0; repetition < numOfFoldRepetitions; repetition++) {
				
				AbstractLearningClassifierSystem foldLCS = prototype.createNew(); // foldLCS = new AbstractLearningClassifierSystem
				System.out.println("Training Fold " + i);
				
				//instances.setClassIndex(instances.numAttributes() - numberOfLabels); 
				//instances.stratify(numOfFolds);
								
				try {
					InstancesUtility.splitDatasetIntoFolds(foldLCS, instances, numOfFolds);
				} catch (Exception e1) {
					e1.printStackTrace();
				}
				loadMlStratifiedFold(i, foldLCS);

				//loadFold(i, foldLCS); // mou dinei to trainSet kai to testSet
				
				
				foldLCS.registerMultilabelHooks(InstancesUtility.convertIntancesToDouble(testSet), numberOfLabels);
				
				if (initializePopulation) {
					try {
						foldLCS.setRulePopulation(foldLCS.initializePopulation(trainSet));
						System.out.println("Population initialized.");
					} catch (Exception e) {
						e.printStackTrace();
					}
				}
				
				foldLCS.train();
				
				System.out.println("Label cardinality: " + foldLCS.labelCardinality);

				// Gather results...
				results[repetition] = foldLCS.getEvaluations(testSet);
				// print the results for the current repetition. added 21.09.2012
				printEvaluations(results[repetition]);
				System.out.println("\n");
				
				// log evals to files
				final String[] names = prototype.getEvaluationNames();

				for (int i = 0; i < results[repetition].length; i++) {
					try {
						final FileWriter fstream = new FileWriter(foldLCS.hookedMetricsFileDirectory + "/evals/" + names[i] + ".txt", true);
						final BufferedWriter buffer = new BufferedWriter(fstream);
						buffer.write(String.valueOf(results[repetition][i]));	
						buffer.flush();
						buffer.close();
					} 
					catch (Exception e) {
						e.printStackTrace();
					}	
				} 
 
			}
			
			

			// Determine better repetition
			int best = 0;
			for (int j = 1; j < numOfFoldRepetitions; j++) {
				if (results[j][metricOptimizationIndex] > results[best][metricOptimizationIndex])
					best = j;
			}

			// Gather to fold stats
			gatherResults(results[best], i);  // epistrefei pinaka. 
											  // se ka9e 9esi exei to apotelesma gia ti metriki pou xrisimopoioume.
											  // ka9e 9esi tou kai ena fold.

			

			// mark end of execution
			final Calendar cal_2 = Calendar.getInstance();
			final SimpleDateFormat sdf_2 = new SimpleDateFormat("kk:mm:ss, dd/MM/yyyy");
			String timestampStop = sdf_2.format(cal_2.getTime());
			System.out.println("\nExecution stopped @ " + timestampStop + "\n");
			
			
		}

		/**
		 * Load a fold into the evaluator.
		 * 
		 * @param foldNumber
		 *            the fold's index
		 * @param lcs
		 *            the LCS that will be used to evaluate this fold
		 */
		private void loadFold(int foldNumber,
				AbstractLearningClassifierSystem lcs) {

			trainSet = instances.trainCV(numOfFolds, foldNumber);
			lcs.instances = InstancesUtility.convertIntancesToDouble(trainSet);
			testSet = instances.testCV(numOfFolds, foldNumber);
			lcs.labelCardinality = InstancesUtility.getLabelCardinality(trainSet); // ?? really ?? i forgot THIS ??
		}
		
		/**
		 * Load a multilabel stratified fold into the evaluator.
		 * 
		 * @param foldNumber
		 *            the fold's index
		 * @param lcs
		 *            the LCS that will be used to evaluate this fold
		 */
		private void loadMlStratifiedFold(int foldNumber,
											AbstractLearningClassifierSystem lcs) {
			
			Instances trainInstances = new Instances (instances, 0);
			Instances testInstances = new Instances (instances, 0);
			
			// to mege9os tou testInstances vector einai diplasio apo ton ari9mo ton instances logo tou pos douleuei i me9odos add (int, instance)
			// opote sto telos einai gemato me adeious pinakes logo placeholding 
			int numberOfPartitions = InstancesUtility.testInstances.size() / 2;
			
			for (int i = 0; i < numberOfPartitions; i++) {
				for (int j = 0; j < InstancesUtility.testInstances.elementAt(i)[foldNumber].numInstances(); j++) {
					testInstances.add(InstancesUtility.testInstances.elementAt(i)[foldNumber].instance(j));
					
				}
				for (int j = 0; j < InstancesUtility.trainInstances.elementAt(i)[foldNumber].numInstances(); j++) {
					trainInstances.add(InstancesUtility.trainInstances.elementAt(i)[foldNumber].instance(j));
					
				}
				
			}
			
			trainSet = trainInstances;
			lcs.instances = InstancesUtility.convertIntancesToDouble(trainSet);
			testSet = testInstances;
			lcs.labelCardinality = InstancesUtility.getLabelCardinality(trainSet);
			
			
			//System.out.println("trainset: \n" + trainSet);
			//System.out.println("testSet: \n" + testSet);

		}
	}

	/**
	 * The number of folds to separate the dataset.
	 * @uml.property  name="numOfFolds"
	 */
	private final int numOfFolds;

	/**
	 * The LCS prototype to be evaluated.
	 * @uml.property  name="prototype"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	private final AbstractLearningClassifierSystem prototype;

	/**
	 * The instances that the LCS will be evaluated on.
	 * @uml.property  name="instances"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	private final Instances instances;

	/**
	 * The evaluations.
	 * @uml.property  name="evals" multiplicity="(0 -1)" dimension="2"
	 */
	private double[][] evals;

	/**
	 * The runs to run.
	 * @uml.property  name="runs"
	 */
	final int runs;

	/**
	 * An executor service containing a thread pool to run folds
	 * @uml.property  name="threadPool"
	 */
	final ExecutorService threadPool;

	/**
	 * Constructor.
	 * 
	 * @param folds
	 *            the number of folds
	 * @param myLcs
	 *            the LCS instance to be evaluated
	 * @param filename
	 *            the filename of the .arff containing the instances
	 * @throws IOException
	 *             if the file is not found.
	 */
	public FoldEvaluator(int folds, AbstractLearningClassifierSystem myLcs,
			final String filename) throws IOException {
		numOfFolds = folds;
		prototype = myLcs; 

		instances = InstancesUtility.openInstance(filename);
		runs = (int) SettingsLoader.getNumericSetting("foldsToRun", numOfFolds);
		instances.randomize(new Random());
		int numOfThreads = (int) SettingsLoader.getNumericSetting("numOfThreads", 1);
		threadPool = Executors.newFixedThreadPool(numOfThreads);

	}

	/**
	 * Constructor.
	 * 
	 * @param folds
	 *            the number of folds used at evaluation
	 * @param numberOfRuns
	 *            the number of runs
	 * @param myLcs
	 *            the LCS under evaluation
	 * @param inputInstances
	 *            the instances to evaluate the LCS on
	 */
	public FoldEvaluator(int folds, int numberOfRuns,
			AbstractLearningClassifierSystem myLcs, Instances inputInstances) {
		numOfFolds = folds;
		prototype = myLcs;
		instances = inputInstances;
		runs = numberOfRuns;

		int numOfThreads = (int) SettingsLoader.getNumericSetting(
				"numOfThreads", 1);
		threadPool = Executors.newFixedThreadPool(numOfThreads);
	}

	/**
	 * Calculate the mean of all fold metrics.
	 * 
	 * @param results
	 *            the results double array
	 * @return the mean for each row
	 */
	public double[] calcMean(double[][] results) {
		final double[] means = new double[results[0].length];
		for (int i = 0; i < means.length; i++) {
			double sum = 0;
			for (int j = 0; j < results.length; j++) {
				sum += results[j][i];
			}
			means[i] = (sum) / (results.length);
		}
		return means;
	}

	/**
	 * Perform evaluation.
	 */
	public void evaluate() {
		
		
		final int metricOptimizationIndex = (int) SettingsLoader.getNumericSetting("metricOptimizationIndex", 0);
		final int numOfFoldRepetitions = (int) SettingsLoader.getNumericSetting("numOfFoldRepetitions", 1); // repeat process per fold

		// kalei ti run() {runs} fores
		for (int currentRun = 0; currentRun < runs; currentRun++) {
			Runnable foldEval = new FoldRunnable(metricOptimizationIndex, currentRun, numOfFoldRepetitions);
			this.threadPool.execute(foldEval);
		}

		this.threadPool.shutdown();
		try {
			this.threadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);
		} catch (InterruptedException e) {
			System.out.println("Thread Pool Interrupted");
			e.printStackTrace();
		}
		final double[] means = calcMean(this.evals);
		// print results
		printEvaluations(means);
		

	}

	/**
	 * Gather the results from a specific fold.
	 * 
	 * @param results
	 *            the results array
	 * @param fold
	 *            the fold the function is currently gathering
	 * 
	 * @return the double containing all evaluations (up to the point being
	 *         added)
	 */
	public synchronized double[][] gatherResults(double[] results, int fold) {
		if (evals == null) {
			evals = new double[runs][results.length];
		}

		evals[fold] = results;

		return evals;

	}

	/**
	 * Print the evaluations.
	 * 
	 * @param means
	 *            the array containing the evaluation means
	 */
	public void printEvaluations(double[] means) {
		final String[] names = prototype.getEvaluationNames();

		for (int i = 0; i < means.length; i++) {
			System.out.println(names[i] + ": " + means[i]);
			if ((i + 1) % 4 == 0) System.out.println();
		}  
	}

}
