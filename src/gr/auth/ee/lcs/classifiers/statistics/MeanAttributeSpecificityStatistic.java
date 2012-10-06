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
package gr.auth.ee.lcs.classifiers.statistics;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.classifiers.Classifier;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.data.ClassifierTransformBridge;
import gr.auth.ee.lcs.data.ILCSParallelMetric;

import edu.rit.pj.ParallelRegion;
import edu.rit.pj.ParallelTeam;
import edu.rit.pj.IntegerForLoop;
import edu.rit.pj.ParallelSection;


/**
 * Calculate the mean attribute specificity statistic.
 * 
 * @author Miltiadis Allamanis
 * 
 */


public class MeanAttributeSpecificityStatistic implements ILCSParallelMetric {

	/*
	 * (non-Javadoc)
	 * 
	 * @see gr.auth.ee.lcs.data.ILCSMetric#getMetric(gr.auth.ee.lcs.
	 * AbstractLearningClassifierSystem)
	 */
	
	private static ClassifierSet set;
	private static int numberOfAttributes;
	private static ClassifierTransformBridge bridge;
	private static int numberOfMacroclassifiers;
	private static int specificAttributes;
		
	@Override
	public double getMetric(AbstractLearningClassifierSystem lcs) {
				
		final ClassifierTransformBridge bridge = lcs.getClassifierTransformBridge();
		final int numberOfAttributes = bridge.getNumberOfAttributes();
		final ClassifierSet set = lcs.getRulePopulation();
		final int numberOfMacroclassifiers = set.getNumberOfMacroclassifiers();

		int specificAttibutes = 0;
		for (int i = 0; i < numberOfMacroclassifiers; i++) {
			final Classifier cl = set.getClassifier(i);
			final int numerosity = set.getClassifierNumerosity(i);
			for (int j = 0; j < numberOfAttributes; j++) {
				if (bridge.isAttributeSpecific(cl, j)) {
					specificAttibutes += numerosity;
				}
			}
		}

		return ((double) specificAttibutes)
				/ ((double) (numberOfMacroclassifiers * Math.pow(numberOfAttributes, 2)));
	}
	
	@Override
	public double getSmpMetric( AbstractLearningClassifierSystem lcs )
	{
		bridge = lcs.getClassifierTransformBridge();
		numberOfAttributes = bridge.getNumberOfAttributes();
		set = lcs.getRulePopulation();
		numberOfMacroclassifiers = set.getNumberOfMacroclassifiers();
				
		try{
		new ParallelTeam().execute(new ParallelRegion()
		{
			public void start()
			{
				specificAttributes = 0;
			}
			public void run() throws Exception
			{
				execute(0,numberOfMacroclassifiers-1,new IntegerForLoop()
				{	
					int specificAttributesLocal = 0;
					
					public void run(int first,int last)
					{
						for (int i = first ; i <= last ; i++)
						{
							Classifier cl  = set.getClassifier(i);
							int numerosity = set.getClassifierNumerosity(i);
							for ( int j = 0 ; j < numberOfAttributes ; j++ )
							{
								if( bridge.isAttributeSpecific(cl, j) )
								{
									specificAttributesLocal += numerosity;
								}
							}
						}
					}
					public void finish() throws Exception
					{
						region().critical(new ParallelSection()
						{
							public void run() 
							{
								specificAttributes += specificAttributesLocal;
							}
						});
					}
				});
			}			
		}); 
		}
		catch (Exception e)
		{
			return getMetric(lcs);
		}
		
		return (double) (specificAttributes)/ ((double) (numberOfMacroclassifiers * Math.pow(numberOfAttributes, 2)));
	
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see gr.auth.ee.lcs.data.ILCSMetric#getMetricName()
	 */
	@Override
	public String getMetricName() {
		return "Mean Attribute Specificity";
	}
	
	@Override
	public double getMetric( AbstractLearningClassifierSystem lcs, boolean smp)
	{
		if (smp == true)
			return getSmpMetric(lcs);
		else
			return getMetric(lcs);
	}

}