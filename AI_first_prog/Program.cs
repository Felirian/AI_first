using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace TransferLearningTF
{
    class Program
    {
        // <SnippetDeclareGlobalVariables>
        static readonly string _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
        static readonly string _imagesFolder = Path.Combine(_assetsPath, "images");
        static readonly string _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
        static readonly string _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv");
        static readonly string _predictSingleImage = Path.Combine(_imagesFolder, "toaster3.jpg");
        static readonly string _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");
        // </SnippetDeclareGlobalVariables>

        static void Main(string[] args)
        {           
            MLContext mlContext = new MLContext();
            
            ITransformer model = GenerateModel(mlContext);            
            ClassifySingleImage(mlContext, model);
            
            Console.ReadKey();
        }
        
        public static ITransformer GenerateModel(MLContext mlContext)
        {            
            IEstimator<ITransformer> pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _imagesFolder, inputColumnName: nameof(ImageData.ImagePath))
                            
                            .Append(mlContext.Transforms.ResizeImages(
                                outputColumnName: "input", 
                                imageWidth: InceptionSettings.ImageWidth, 
                                imageHeight: InceptionSettings.ImageHeight, 
                                inputColumnName: "input"
                                )
                            )
                            
                            .Append(mlContext.Transforms.ExtractPixels(
                                outputColumnName: "input", 
                                interleavePixelColors: InceptionSettings.ChannelsLast, 
                                offsetImage: InceptionSettings.Mean
                                )
                            )
                            
                            
                            .Append(mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel).
                                ScoreTensorFlowModel(outputColumnNames: 
                                    new[] { "softmax2_pre_activation" }, 
                                    inputColumnNames: new[] { "input" }, 
                                    addBatchDimensionInput: true
                                ))
                            
                            .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
                            
                            .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
                                labelColumnName: "LabelKey", 
                                featureColumnName: "softmax2_pre_activation"
                                )
                            )
                            
                            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
                            .AppendCacheCheckpoint(mlContext);
          
            IDataView trainingData = mlContext.Data.LoadFromTextFile<ImageData>(path: _trainTagsTsv, hasHeader: false);
           
            Console.WriteLine("=============== Training classification model ===============");
           
            ITransformer model = pipeline.Fit(trainingData);
            
            IDataView testData = mlContext.Data.LoadFromTextFile<ImageData>(path: _testTagsTsv, hasHeader: false);
            IDataView predictions = model.Transform(testData);

            IEnumerable<ImagePrediction> imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);
            DisplayResults(imagePredictionData);
          
            Console.WriteLine("=============== Classification metrics ===============");
          
            MulticlassClassificationMetrics metrics =
                mlContext.MulticlassClassification.Evaluate(predictions,
                  labelColumnName: "LabelKey",
                  predictedLabelColumnName: "PredictedLabel");
          
            Console.WriteLine($"Ошибки: {metrics.LogLoss}");
            Console.WriteLine($"ошибки по классам: {String.Join(" , \n ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");
            
            return model;            
        }

        public static void ClassifySingleImage(MLContext mlContext, ITransformer model)
        {
            var imageData = new ImageData()
            {
                ImagePath = _predictSingleImage
            };
           
            var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var prediction = predictor.Predict(imageData);
           
            Console.WriteLine("=============== Making single image classification ===============");
          
            Console.WriteLine($"картинка: {Path.GetFileName(imageData.ImagePath)} класс: {prediction.PredictedLabelValue} вероятность: {prediction.Score.Max()} ");            
        }

        private static void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
        {
            foreach (ImagePrediction prediction in imagePredictionData)
            {
                Console.WriteLine($"Картинка: {Path.GetFileName(prediction.ImagePath)} класс: {prediction.PredictedLabelValue} вероятность: {prediction.Score.Max()} ");
            }           
        }

        private struct InceptionSettings
        {
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 150;
            public const float Scale = 1;
            public const bool ChannelsLast = true;
        }
        
        public class ImageData
        {
            [LoadColumn(0)]
            public string ImagePath;

            [LoadColumn(1)]
            public string Label;
        }
        
        public class ImagePrediction : ImageData
        {
            public float[] Score;

            public string PredictedLabelValue;
        }        
    }
}
