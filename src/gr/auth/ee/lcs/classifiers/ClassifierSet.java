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
package gr.auth.ee.lcs.classifiers;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Collections;
import java.util.Vector;
import java.text.*;
/**
 * Implement set of Classifiers, counting numerosity for classifiers. This
 * object is serializable.
 * 
 * @author Miltos Allamanis
 * 
 * @has 1 - * Macroclassifier
 * @has 1 - 1 IPopulationControlStrategy
 */
public class ClassifierSet implements Serializable {

	/**
	 * Serialization id for versioning.
	 */
	private static final long serialVersionUID = 2664983888922912954L;
	
	public int totalGAInvocations = 0;

	/**
	 * Open a saved (and serialized) ClassifierSet.
	 * 
	 * @param path
	 *            the path of the ClassifierSet to be opened
	 * @param sizeControlStrategy
	 *            the ClassifierSet's
	 * @param lcs
	 *            the lcs which the new set will belong to
	 * @return the opened classifier set
	 */
	public static ClassifierSet openClassifierSet(final String path,
			final IPopulationControlStrategy sizeControlStrategy,
			final AbstractLearningClassifierSystem lcs) {
		FileInputStream fis = null;
		ObjectInputStream in = null;
		ClassifierSet opened = null;

		try {
			fis = new FileInputStream(path);
			in = new ObjectInputStream(fis);

			opened = (ClassifierSet) in.readObject();
			opened.myISizeControlStrategy = sizeControlStrategy;

			for (int i = 0; i < opened.getNumberOfMacroclassifiers(); i++) {
				final Classifier cl = opened.getClassifier(i);
				cl.setLCS(lcs);
			}

			in.close();
		} catch (IOException ex) {
			ex.printStackTrace();
		} catch (ClassNotFoundException ex) {
			ex.printStackTrace();
		}

		return opened;
	}

	/**
	 * A static function to save the classifier set.
	 * 
	 * @param toSave
	 *            the set to be saved
	 * @param filename
	 *            the path to save the set
	 */
	public static void saveClassifierSet(final ClassifierSet toSave,
			final String filename) {
		FileOutputStream fos = null;
		ObjectOutputStream out = null;

		try {
			fos = new FileOutputStream(filename);
			out = new ObjectOutputStream(fos);
			out.writeObject(toSave);
			out.close();

		} catch (IOException ex) {
			ex.printStackTrace();
		}

	}

	/**
	 * The total numerosity of all classifiers in set.
	 * @uml.property  name="totalNumerosity"
	 */
	private int totalNumerosity = 0;

	/**
	 * Macroclassifier vector.
	 * @uml.property  name="myMacroclassifiers"
	 * @uml.associationEnd  multiplicity="(0 -1)" elementType="gr.auth.ee.lcs.classifiers.Macroclassifier"
	 */
	private final Vector<Macroclassifier> myMacroclassifiers;

	/**
	 * An interface for a strategy on deleting classifiers from the set. This attribute is transient and therefore not serializable.
	 * @uml.property  name="myISizeControlStrategy"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	private transient IPopulationControlStrategy myISizeControlStrategy;

	/**
	 * The default ClassifierSet constructor.
	 * 
	 * @param sizeControlStrategy
	 *            the size control strategy to use for controlling the set
	 */
	public ClassifierSet(final IPopulationControlStrategy sizeControlStrategy) {
		this.myISizeControlStrategy = sizeControlStrategy;
		this.myMacroclassifiers = new Vector<Macroclassifier>();

	}

	/**
	 * Adds a classifier with the a given numerosity to the set. It checks if
	 * the classifier already exists and increases its numerosity. It also
	 * checks for subsumption and updates the set's numerosity.
	 * 
	 * @param thoroughAdd
	 *            to thoroughly check addition
	 * @param macro
	 *            the macroclassifier to add to the set
	 */
	public final void addClassifier(final Macroclassifier macro,
									  final boolean thoroughAdd) {

		final int numerosity = macro.numerosity;
		// Add numerosity to the Set
		this.totalNumerosity += numerosity;

		// Subsume if possible
		if (thoroughAdd) {
			final Classifier aClassifier = macro.myClassifier;
			for (int i = 0; i < myMacroclassifiers.size(); i++) {
				
				final Classifier theClassifier = myMacroclassifiers.elementAt(i).myClassifier;
				
				if (theClassifier.canSubsume()) {
					if (theClassifier.isMoreGeneral(aClassifier)) {
						// Subsume and control size...
						myMacroclassifiers.elementAt(i).numerosity += numerosity;
						myMacroclassifiers.elementAt(i).numberOfSubsumptions++;

						if (myISizeControlStrategy != null) {
							myISizeControlStrategy.controlPopulation(this);
						}
						return;
					}
				} else if (theClassifier.equals(aClassifier)) { // Or it can't
																// subsume but
																// it is equal
					
					myMacroclassifiers.elementAt(i).numerosity += numerosity;
					myMacroclassifiers.elementAt(i).numberOfSubsumptions++;

					if (myISizeControlStrategy != null) {
						myISizeControlStrategy.controlPopulation(this);
					}
					return;
				}
			}
		}

		/*
		 * No matching or subsumable more general classifier found. Add and
		 * control size...
		 * 
		 * pros9ese ton macroclassifier sto vector myMacroclassifiers.
		 * sti sunexeia an exei oristei stratigiki diagrafis, ektelese tin.
		 * an to numerocity ton macroclassifiers einai pano apo to populationSize arxise na diagrafeis
		 * 
		 * 
		 * an borei na kanei subsume de 9a ektelesei tis parakato entoles (return statements pio pano)
		 */
		this.myMacroclassifiers.add(macro);
		if (myISizeControlStrategy != null) {
			myISizeControlStrategy.controlPopulation(this);
		}
	}

	/**
	 * Removes a micro-classifier from the set. It either completely deletes it
	 * (if the classsifier's numerosity is 0) or by decreasing the numerosity.
	 * 
	 * @param aClassifier
	 *            the classifier to delete
	 */
	public final void deleteClassifier(final Classifier aClassifier) {

		int index;
		final int macroSize = myMacroclassifiers.size();
		for (index = 0; index < macroSize; index++) {
			if (myMacroclassifiers.elementAt(index).myClassifier.getSerial() ==  aClassifier.getSerial()) {
				break;
			}
		}

		if (index == macroSize)
			return;
		deleteClassifier(index);

	}

	/**
	 * Deletes a classifier with the given index. If the macroclassifier at the
	 * given index contains more than one classifier the numerosity is decreased
	 * by one.
	 * 
	 * @param index
	 *            the index of the classifier's macroclassifier to delete
	 */
	public final void deleteClassifier(final int index) {
		
		this.totalNumerosity--; // meiose to numerosity olou tou set
		if (this.myMacroclassifiers.elementAt(index).numerosity > 1) {
			this.myMacroclassifiers.elementAt(index).numerosity--; 
		} else {
			this.myMacroclassifiers.remove(index); // an to numerosity tou macroclassifier einai 1, diagrapse ton
		}
	}

	
	
	/**
	 * It completely removes the macroclassifier with the given index.
	 * Used by cleanUpZeroCoverage().
	 * 
	 * @param index
	 * 
	 * @author alexandros filotheou
	 * 
	 * */
	
	
	public final void deleteMacroclassifier (final int index) {
		
		this.totalNumerosity -= this.myMacroclassifiers.elementAt(index).numerosity;
		this.myMacroclassifiers.remove(index);
	
	}
	
	
	
	
	/**
	 * Generate a match set for a given instance.
	 * 
	 * @param dataInstance
	 *            the instance to be matched
	 * @return a ClassifierSet containing the match set
	 */
	public final ClassifierSet generateMatchSet(final double[] dataInstance) {
		final ClassifierSet matchSet = new ClassifierSet(null);
		final int populationSize = this.getNumberOfMacroclassifiers();
		// TODO: Parallelize for performance increase
		for (int i = 0; i < populationSize; i++) {
			if (this.getClassifier(i).isMatch(dataInstance)) {
				matchSet.addClassifier(this.getMacroclassifier(i), false);
			}
		}
		return matchSet;
	}

	/**
	 * Generate match set from data instance.
	 * 
	 * @param dataInstanceIndex
	 *            the index of the instance
	 * @return the match set
	 */
	public final ClassifierSet generateMatchSet(final int dataInstanceIndex) {
		
		final ClassifierSet matchSet = new ClassifierSet(null); // kataskeuazoume ena adeio arxika classifierSet
		final int populationSize = this.getNumberOfMacroclassifiers();
		// TODO: Parallelize for performance increase
		for (int i = 0; i < populationSize; i++) {
			
			// this = population (macroclassifiers)
			// apo tous macroclassifiers pou sun9etoun ton plh9usmo pairno autous pou einai match me to vision vector
			// getClassifier(i) <--- this.myMacroclassifiers.elementAt(index).myClassifier;
			

			if (this.getClassifier(i).isMatch(dataInstanceIndex)) { 
				
				matchSet.addClassifier(this.getMacroclassifier(i), false); 
			}
			
			boolean zeroCoverage = (this.getClassifier(i).getCheckedInstances() >= this.getClassifier(i).getLCS().instances.length) 
									 && (this.getClassifier(i).getCoverage() == 0);
			
			if(zeroCoverage) {
				//matchSet.deleteMacroclassifier(i);
				this.deleteMacroclassifier(i);
			}
		}
		return matchSet;
	}

	/**
	 * Return the classifier at a given index of the macroclassifier vector.
	 * 
	 * @param index
	 *            the index of the macroclassifier
	 * @return the classifier at the specified index
	 */
	public final Classifier getClassifier(final int index) {
		return this.myMacroclassifiers.elementAt(index).myClassifier;
	}

	/**
	 * Returns a classifier's numerosity (the number of microclassifiers).
	 * 
	 * @param aClassifier
	 *            the classifier
	 * @return the given classifier's numerosity
	 */
	public final int getClassifierNumerosity(final Classifier aClassifier) {
		for (int i = 0; i < myMacroclassifiers.size(); i++) {
			if (myMacroclassifiers.elementAt(i).myClassifier.getSerial() == aClassifier
					.getSerial()) // ka9e (micro)classifier exei kai ena serial number. 
								  // apo oti exo katalabei diaforetiko gia ka9e microclassifier, 
								  // akoma kai gia autous tou idiou macroclassifier
				//System.out.println("myClassifier" + myMacroclassifiers.elementAt(i).myClassifier.getSerial());
				//System.out.println(aClassifier.getSerial());
				return this.myMacroclassifiers.elementAt(i).numerosity;
		}
		return 0;
	}

	/**
	 * Overloaded function for getting a numerosity.
	 * 
	 * @param index
	 *            the index of the macroclassifier
	 * @return the index'th macroclassifier numerosity
	 */
	public final int getClassifierNumerosity(final int index) {
		return this.myMacroclassifiers.elementAt(index).numerosity;
	}

	/**
	 * Returns (a copy of) the macroclassifier at the given index.
	 * 
	 * @param index
	 *            the index of the macroclassifier vector
	 * @return the macroclassifier at a given index
	 */
	public final Macroclassifier getMacroclassifier(final int index) {
		return new Macroclassifier(this.myMacroclassifiers.elementAt(index));
		//return this.myMacroclassifiers.elementAt(index);
	}
	
	
	/**
	 * Returns the actual (not a copy as the aboce method) macroclassifier at the given index.
	 * 
	 * @param index
	 *            the index of the macroclassifier vector
	 * @return the macroclassifier at a given index
	 * 
	 * @author alexandros filotheou
	 */
	
	public Macroclassifier getActualMacroclassifier(final int index) {
		return this.myMacroclassifiers.elementAt(index);
	}

	/**
	 * Getter.
	 * 
	 * @return the number of macroclassifiers in the set
	 */
	public final int getNumberOfMacroclassifiers() {
		return this.myMacroclassifiers.size();
	}

	/**
	 * Get the set's population control strategy
	 * 
	 * @return the set's population control strategy
	 */
	public final IPopulationControlStrategy getPopulationControlStrategy() {
		return myISizeControlStrategy;
	}

	/**
	 * Returns the set's total numerosity (the total number of microclassifiers).
	 * @return  the sets total numerosity
	 * @uml.property  name="totalNumerosity"
	 */
	public final int getTotalNumerosity() {
		return this.totalNumerosity;
	}

	/**
	 * @return true if the set is empty
	 */
	public final boolean isEmpty() {
		return this.myMacroclassifiers.isEmpty();
	}

	/**
	 * Merge a set into this set.
	 * 
	 * @param aSet
	 *            the set to be merged.
	 */
	public final void merge(final ClassifierSet aSet) {
		final int setSize = aSet.getNumberOfMacroclassifiers();
		for (int i = 0; i < setSize; i++) {
			final Macroclassifier ml = aSet.getMacroclassifier(i);
			this.addClassifier(ml, false);
		}
	}

	/**
	 * Print all classifiers in the set.
	 */
	public final void print() {
		System.out.println(toString());
	}

	/**
	 * Remove all set's macroclassifiers.
	 */
	public final void removeAllMacroclassifiers() {
		this.myMacroclassifiers.clear();
		this.totalNumerosity = 0;
	}

	/**
	 * Self subsume. the fuck?
	 */
	public final void selfSubsume() {
		for (int i = 0; i < this.getNumberOfMacroclassifiers(); i++) {
			final Macroclassifier cl = this.getMacroclassifier(0);
			final int numerosity = cl.numerosity;
			this.myMacroclassifiers.remove(0);
			this.totalNumerosity -= numerosity;
			this.addClassifier(cl, true);
		}
	}

	/*
	 * tou miltou, peiragmenos mexri enos simeiou
	 * 
	 * @Override
	public String toString() {
		final StringBuffer response = new StringBuffer();
		
		double numOfCover = 0;
		double numOfGA = 0;
		int numOfSubsumptions = 0;
		
		//Macroclassifier sortedMacroClassifiers [] = new Macroclassifier [this.getNumberOfMacroclassifiers()];
		
		for (int i = 0; i < this.getNumberOfMacroclassifiers(); i++) {
			
			myMacroclassifiers.elementAt(i).totalFitness = this.getClassifier(i).getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLOITATION) * this.getMacroclassifier(i).numerosity;
			System.out.println("AAAAAA"+myMacroclassifiers.elementAt(i).totalFitness);
		}
		
		Collections.sort(myMacroclassifiers);

			
		for (int i = 0; i < this.getNumberOfMacroclassifiers(); i++) {
			
			//response.append(this.getClassifier(i).toString()
			response.append(myMacroclassifiers.elementAt(i).myClassifier.toString()
					+ " fit:" + this.getClassifier(i).getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLOITATION)
					//+ " total fitness: " + this.getClassifier(i).getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLOITATION) * this.getMacroclassifier(i).numerosity
					+ " total fitness: " + myMacroclassifiers.elementAt(i).totalFitness
					+ " exp:" + this.getClassifier(i).experience 
					+ " num:" + this.getClassifierNumerosity(i) 
					+ " cov:" + this.getClassifier(i).getCoverage()
					+ System.getProperty("line.separator"));
			
			response.append(this.getClassifier(i).getUpdateSpecificData()
					+ System.getProperty("line.separator"));
			
			if (this.getClassifier(i).getClassifierOrigin() == "cover") {
				numOfCover++;
				response.append(" origin: cover "); }
			else if (this.getClassifier(i).getClassifierOrigin() == "ga") {
				numOfGA++;
				response.append(" origin: ga "); 
			}
			
			numOfSubsumptions += this.getMacroclassifier(i).numberOfSubsumptions;


			response.append(" created: " + this.getClassifier(i).timestamp + " ");
			
		}
		
		
		System.out.println("Population size:" + this.getNumberOfMacroclassifiers());
		System.out.println("Number of classifiers covered:" + (int) numOfCover);
		System.out.println("Number of classifiers ga-ed:" + (int) numOfGA);
		
		System.out.println("% covered:" + 100 * (numOfCover / this.getNumberOfMacroclassifiers()) + " %");
		System.out.println("% ga-ed:" + 100 * (numOfGA / this.getNumberOfMacroclassifiers()) + " %");
		
		//System.out.println("Total number of epochs:" + this.getClassifier(this.getNumberOfMacroclassifiers() - 1).timestamp);
		System.out.println("Total number of epochs:" + this.totalGAInvocations);

		System.out.println("Total number of subsumptions:" + numOfSubsumptions);

		return response.toString();
	}*/
	
	
/*	@Override
	public String toString() {
		final StringBuffer response = new StringBuffer();
		
		double numOfCover = 0;
		double numOfGA = 0;
		int numOfSubsumptions = 0;
		
		//Macroclassifier sortedMacroClassifiers [] = new Macroclassifier [this.getNumberOfMacroclassifiers()];
		
		for (int i = 0; i < this.getNumberOfMacroclassifiers(); i++) {
			myMacroclassifiers.elementAt(i).totalFitness = this.getClassifier(i).getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLOITATION) * this.getMacroclassifier(i).numerosity;
		}
		
		Collections.sort(myMacroclassifiers);

			
		for (int i = 0; i < this.getNumberOfMacroclassifiers(); i++) {
			
			//response.append(this.getClassifier(i).toString()
			response.append(myMacroclassifiers.elementAt(i).myClassifier.toString()
					+ " fit:" + myMacroclassifiers.elementAt(i).myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLOITATION)
					//+ " total fitness: " + this.getClassifier(i).getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLOITATION) * this.getMacroclassifier(i).numerosity
					+ " total fitness: " + myMacroclassifiers.elementAt(i).totalFitness
					+ " exp:" + myMacroclassifiers.elementAt(i).myClassifier.experience 
					+ " num:" + myMacroclassifiers.elementAt(i).numerosity
					+ " cov:" + myMacroclassifiers.elementAt(i).myClassifier.getCoverage()
					+ System.getProperty("line.separator"));
			
			response.append(myMacroclassifiers.elementAt(i).myClassifier.getUpdateSpecificData()
					+ System.getProperty("line.separator"));
			
			if (myMacroclassifiers.elementAt(i).myClassifier.getClassifierOrigin() == "cover") {
				numOfCover++;
				response.append(" origin: cover "); }
			else if (myMacroclassifiers.elementAt(i).myClassifier.getClassifierOrigin() == "ga") {
				numOfGA++;
				response.append(" origin: ga "); 
			}
			
			numOfSubsumptions += myMacroclassifiers.elementAt(i).numberOfSubsumptions;


			response.append(" created: " + myMacroclassifiers.elementAt(i).myClassifier.timestamp + " ");
			
		}
		
		
		System.out.println("Population size:" + this.getNumberOfMacroclassifiers());
		System.out.println("Number of classifiers covered:" + (int) numOfCover);
		System.out.println("Number of classifiers ga-ed:" + (int) numOfGA);
		
		System.out.println("% covered:" + 100 * (numOfCover / this.getNumberOfMacroclassifiers()) + " %");
		System.out.println("% ga-ed:" + 100 * (numOfGA / this.getNumberOfMacroclassifiers()) + " %");
		
		//System.out.println("Total number of epochs:" + this.getClassifier(this.getNumberOfMacroclassifiers() - 1).timestamp);
		System.out.println("Total number of epochs:" + this.totalGAInvocations);

		System.out.println("Total number of subsumptions:" + numOfSubsumptions);

		return response.toString();
	}*/
	
	@Override
	public String toString() {
		final StringBuffer response = new StringBuffer();
		
		double numOfCover = 0;
		double numOfGA = 0;
		int numOfSubsumptions = 0;
		
		
		for (int i = 0; i < this.getNumberOfMacroclassifiers(); i++) {
			myMacroclassifiers.elementAt(i).totalFitness = 
				this.getClassifier(i).getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLOITATION) * this.getMacroclassifier(i).numerosity;
		}
		
        DecimalFormat df = new DecimalFormat("#.####");

		for (int i = 0; i < this.getNumberOfMacroclassifiers(); i++) {
			
			//response.append(this.getClassifier(i).toString()
			response.append(
						myMacroclassifiers.elementAt(i).myClassifier.toString() // antecedent => concequent
					//+ " total fitness: " + this.getClassifier(i).getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLOITATION) * this.getMacroclassifier(i).numerosity
					// myMacroclassifiers.elementAt(i).toString isos kalutera
					+ " total macro fit: " + df.format(myMacroclassifiers.elementAt(i).totalFitness)
					+ " fit: " + df.format(myMacroclassifiers.elementAt(i).myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLOITATION))
					+ " num: " + myMacroclassifiers.elementAt(i).numerosity
					+ " exp: " + myMacroclassifiers.elementAt(i).myClassifier.experience 
					+ " cov: " + df.format(100 * myMacroclassifiers.elementAt(i).myClassifier.getCoverage()) + "% of dataset");
			
			response.append(myMacroclassifiers.elementAt(i).myClassifier.getUpdateSpecificData());
			
			if (myMacroclassifiers.elementAt(i).myClassifier.getClassifierOrigin() == "cover") {
				numOfCover++;
				response.append(" origin: cover "); 
			}
			else if (myMacroclassifiers.elementAt(i).myClassifier.getClassifierOrigin() == "ga") {
				numOfGA++;
				response.append(" origin: ga "); 
			}
			numOfSubsumptions += myMacroclassifiers.elementAt(i).numberOfSubsumptions;
			response.append(" created: " + myMacroclassifiers.elementAt(i).myClassifier.created + " ");
			response.append(" last in correctset: " + myMacroclassifiers.elementAt(i).myClassifier.timestamp + " ");
			response.append(" subsumptions: " + myMacroclassifiers.elementAt(i).numberOfSubsumptions + " ");
			response.append(System.getProperty("line.separator"));
		}
		
		System.out.println("\nPopulation size:" 							+ this.getNumberOfMacroclassifiers());
		System.out.println("Number of classifiers in population covered :" 	+ (int) numOfCover);
		System.out.println("Number of classifiers in population ga-ed:" 	+ (int) numOfGA);
		
		System.out.println("% covered:" 					+ df.format(100 * (numOfCover / this.getNumberOfMacroclassifiers())) + " %");
		System.out.println("% ga-ed:" 						+ df.format(100 * (numOfGA / this.getNumberOfMacroclassifiers())) + " %");
		
		System.out.println("Total ga invocations:" 							+ this.totalGAInvocations);

		System.out.println("Total subsumptions:"							+ numOfSubsumptions);
		//System.out.println("Total number of epochs:" + this.getClassifier(this.getNumberOfMacroclassifiers() - 1).timestamp);


		return response.toString();
	}

}