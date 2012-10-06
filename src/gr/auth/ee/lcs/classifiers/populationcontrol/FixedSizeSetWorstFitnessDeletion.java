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
package gr.auth.ee.lcs.classifiers.populationcontrol;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.IPopulationControlStrategy;
import gr.auth.ee.lcs.classifiers.Macroclassifier;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.geneticalgorithm.IRuleSelector;

/**
 * A fixed size control strategy. Classifiers are deleted based on the selector
 * tournaments
 * 
 * @stereotype ConcreteStrategy
 * 
 * @author Miltos Allamanis
 * 
 */
public class FixedSizeSetWorstFitnessDeletion implements
		IPopulationControlStrategy {

	private AbstractLearningClassifierSystem myLcs;
	
	/**
	 * The Natural Selector used to select the the classifier to be deleted.
	 * @uml.property  name="mySelector"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	private final IRuleSelector mySelector;

	/**
	 * The fixed population size of the controlled set.
	 * @uml.property  name="populationSize"
	 */
	private final int populationSize;
	
	private int numberOfDeletions;
	
	private long deletionTime;

	/**
	 * Removes all zero coverage rules
	 * @uml.property  name="zeroCoverageRemoval"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	private InadequeteClassifierDeletionStrategy zeroCoverageRemoval;

	/**
	 * Constructor of deletion strategy.
	 * 
	 * @param maxPopulationSize
	 *            the size that the population will have
	 * @param selector
	 *            the selector used for deleting
	 */
	public FixedSizeSetWorstFitnessDeletion(
											 final AbstractLearningClassifierSystem lcs,
											 final int maxPopulationSize, 
											 final IRuleSelector selector) {
		
		this.populationSize = maxPopulationSize;
		mySelector = selector; // roulette wheel gia ton GMlASLCS3
		zeroCoverageRemoval = new InadequeteClassifierDeletionStrategy(lcs);
		myLcs = lcs;
	}

	/**
	 * @param aSet
	 *            the set to control
	 * @see gr.auth.ee.lcs.classifiers.IPopulationControlStrategy#controlPopulation(gr.auth.ee.lcs.classifiers.ClassifierSet)
	 * 
	 * 
	 * ekteleitai otan kano addClassifier ston population. diladi otan kano cover i ga. (sto cover einai me false to thorough)
	 * diagrapse prota autous pou exoun zero coverage. 
	 * sti sunexeia, an akoma eimaste pano apo to ano orio tou pli9ismou, diagrapse me rouleta osous kanones prepei oste na pesoume kato apo to ano orio
	 */
	@Override
	public final void controlPopulation(final ClassifierSet aSet) {

		final ClassifierSet toBeDeleted = new ClassifierSet(null);
		
//		not necessary anymore. deletion of zero coverage rules occurs right after the formation of the match set
 
/*		if (aSet.getTotalNumerosity() > populationSize) 
			zeroCoverageRemoval.controlPopulation(aSet);*/

		numberOfDeletions = 0;
		deletionTime = 0;
		
		while (aSet.getTotalNumerosity() > populationSize) {
			long time1 = - System.currentTimeMillis();
			
			numberOfDeletions++;
			// se auto to simeio upologizei maxPopulation + 1 pi9anotites, ka9os gia na kli9ei i controlPopulation, prepei na exei uperbei to ano orio tou pli9usmou
			aSet.getMacroclassifiersVector().elementAt(0).myClassifier.getLCS().getUpdateStrategy().computeDeletionProbabilities(aSet);
			mySelector.select(1, aSet, toBeDeleted); // me rouleta
			
			if (toBeDeleted.getClassifier(0).formulaForD == 0) aSet.secondDeletionFormula++;
			else aSet.firstDeletionFormula++;
						
			aSet.deleteClassifier(toBeDeleted.getClassifier(0));
			toBeDeleted.deleteClassifier(0);
			
			time1 += System.currentTimeMillis();
			
			deletionTime += time1;
		}
	}
	
	public final int getNumberOfDeletionsConducted(){
		return numberOfDeletions;
	}
	
	public final long getDeletionTime(){
		return deletionTime;
	}

}
