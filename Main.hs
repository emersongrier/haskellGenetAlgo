
module Main where
import GenetAlgo
import System.Random

--Emerson Grier
--genetic algorithm
-- for the training of a model to classify iris flowers
--see GenetAlgo.hs for algorithm details and function definitions 


main :: IO ()
main = do
    contents <- readFile "iris.data"

    let datalines = lines contents
        spaced = map (map (\c -> if c == ',' then ' ' else c)) datalines
        ddata = map (map read . words) spaced :: [[Double]]
        sdata = init ddata
        lastItems = map last sdata :: [Double]
        answers' = map round lastItems ::[Int]
        modifiedDdata = map init sdata :: [[Double]]

        -- Define initial set of models
        initialModels = makeModels 700 (randomRs ((negate 1.0), 1.0) (mkStdGen 46))

        --Run the forwardPass function iteratively
        numGenerations = 20 :: Int
        finalModels = steps numGenerations initialModels modifiedDdata answers' 1002

        results = map (\model -> rateOnData model modifiedDdata answers') finalModels

        filePath = "model.txt"
        model = head finalModels
        weightsContent = show (getWeights model)
        biasesContent = show (getBiases model)

    writeFile filePath (weightsContent ++ "\n" ++ biasesContent)

    print ("the best model correctly classifies " ++ show (maximum results) ++ " out of 150 flowers")

--700 models
--20 gens
--1.02 mutate up
--0.98 mutate down
-- likelyhood 0.94