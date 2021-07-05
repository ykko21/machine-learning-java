package org.ykko21.ml;

import java.net.URL;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ModelOneVarLoadAndRun {
	public static void main(String[] args) throws Exception{
		ClassLoader loader = ModelOneVarLoadAndRun.class.getClassLoader();
		URL fileURL = loader.getResource("model/ykko_HDF5_format.h5");
		String filePath = fileURL.getPath().replaceFirst("/C:", "C:");		
		MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(filePath);
		System.out.println(model.summary());
		
		INDArray input = null;		
		input = Nd4j.zeros(10, 1);
		input.putScalar(new int[]{0, 0}, 60);
		input.putScalar(new int[]{1, 0}, 64);
		input.putScalar(new int[]{2, 0}, 68);
		input.putScalar(new int[]{3, 0}, 72);
		input.putScalar(new int[]{4, 0}, 76);
		input.putScalar(new int[]{5, 0}, 80);
		input.putScalar(new int[]{6, 0}, 84);
		input.putScalar(new int[]{7, 0}, 88);
		input.putScalar(new int[]{8, 0}, 92);
		input.putScalar(new int[]{9, 0}, 96);
		
		INDArray val = model.output(input);
		System.out.println(val);
	}
}
/*
 * Output in Java

13:28:16.204 [main] WARN org.deeplearning4j.nn.conf.inputs.InputType - Assigning a size of zero. This is normally only valid in model import cases with unknown dimensions.
13:28:16.325 [main] INFO org.nd4j.linalg.factory.Nd4jBackend - Loaded [CpuBackend] backend
13:28:16.330 [main] ERROR org.nd4j.common.config.ND4JClassLoading - Cannot find class [org.nd4j.linalg.jblas.JblasBackend] of provided class-loader.
13:28:16.330 [main] ERROR org.nd4j.common.config.ND4JClassLoading - Cannot find class [org.canova.api.io.data.DoubleWritable] of provided class-loader.
13:28:16.331 [main] ERROR org.nd4j.common.config.ND4JClassLoading - Cannot find class [org.nd4j.linalg.jblas.JblasBackend] of provided class-loader.
13:28:16.332 [main] ERROR org.nd4j.common.config.ND4JClassLoading - Cannot find class [org.canova.api.io.data.DoubleWritable] of provided class-loader.
13:28:16.973 [main] INFO org.nd4j.nativeblas.NativeOpsHolder - Number of threads used for linear algebra: 4
13:28:16.975 [main] INFO org.nd4j.linalg.cpu.nativecpu.CpuNDArrayFactory - Binary level Generic x86 optimization level AVX/AVX2
13:28:17.029 [main] INFO org.nd4j.nativeblas.Nd4jBlas - Number of threads used for OpenMP BLAS: 4
13:28:17.058 [main] INFO org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner - Backend used: [CPU]; OS: [Windows 10]
13:28:17.058 [main] INFO org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner - Cores: [8]; Memory: [2.0GB];
13:28:17.058 [main] INFO org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner - Blas vendor: [OPENBLAS]
13:28:17.062 [main] INFO org.nd4j.linalg.cpu.nativecpu.CpuBackend - Backend build information:
 GCC: "10.3.0"
STD version: 201103L
DEFAULT_ENGINE: samediff::ENGINE_CPU
HAVE_FLATBUFFERS
HAVE_OPENBLAS
13:28:17.147 [main] INFO org.deeplearning4j.nn.multilayer.MultiLayerNetwork - Starting MultiLayerNetwork with WorkspaceModes set to [training: ENABLED; inference: ENABLED], cacheMode set to [NONE]

======================================================================
LayerName (LayerType)      nIn,nOut   TotalParams   ParamsShape       
======================================================================
dense_2 (DenseLayer)       1,10       20            W:{1,10}, b:{1,10}
dense_3 (DenseLayer)       10,1       11            W:{10,1}, b:{1,1} 
dense_3_loss (LossLayer)   -,-        0             -                 
----------------------------------------------------------------------
            Total Parameters:  31
        Trainable Parameters:  31
           Frozen Parameters:  0
======================================================================

[[70.5522], 
 [75.1399], 
 [79.7276], 
 [84.3154], 
 [88.9031], 
 [93.4908], 
 [98.0785], 
 [102.6663], 
 [107.2540], 
 [111.8417]]
*/


/*
Output in Python
(<tf.Tensor: shape=(10,), dtype=int32, numpy=array([60, 64, 68, 72, 76, 80, 84, 88, 92, 96], dtype=int32)>,
 array([[ 70.55218 ],
        [ 75.13991 ],
        [ 79.72763 ],
        [ 84.31535 ],
        [ 88.903076],
        [ 93.49081 ],
        [ 98.07853 ],
        [102.66625 ],
        [107.253975],
        [111.8417  ]], dtype=float32))
 */

//Reference: https://towardsdatascience.com/deploying-keras-deep-learning-models-with-java-62d80464f34a
