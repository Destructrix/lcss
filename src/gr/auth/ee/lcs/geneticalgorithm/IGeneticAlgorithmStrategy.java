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
package gr.auth.ee.lcs.geneticalgorithm;

import java.util.Vector;

import gr.auth.ee.lcs.classifiers.ClassifierSet;

/**
 * An interface for evolving a set.
 * 
 * @author Miltos Allamanis
 */
public interface IGeneticAlgorithmStrategy {

	/**
	 * An interface for the different strategies for genetically evolving a
	 * population.
	 * 
	 * @param evolveSet
	 *            The set to evolve
	 * @param population
	 *            The population to add new classifiers
	 */
	void evolveSet(ClassifierSet evolveSet, ClassifierSet population);
	
	int getTimestamp(); // @author alek

	
	void evolveSetNew(ClassifierSet evolveSet, ClassifierSet population);
	
	int getActivationAge();
	
	int getNumberOfSubsumptionsConducted();
	
	int getNumberOfNewClassifiers();
		
	void increaseTimestamp();
	
	int getMeanAge(ClassifierSet evolveSet);
	
	int evolutionConducted();
	
	Vector<Integer> getIndicesToSubsume();
	
	ClassifierSet getNewClassifiersSet();
	
	long getSubsumptionTime();
	
	int getNumberOfDeletionsConducted();
	
	long getDeletionTime();
	
	
}