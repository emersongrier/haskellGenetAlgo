module GenetAlgo where

import System.Random
import Data.List
import Data.Maybe (fromJust, fromMaybe)
import Data.Ord

--Emerson Grier
--genetic algorithm,
-- for the training of a model to classify iris flowers

{-
The general algorithm is

    1.) create randomly generated population of models
        --these models are represented as single layer neuralnetwork
    
    2.) then iteratively:
        --the models are then tested against the data, and rated
        --the best 50% of models reproduce by having their paramters shuffled into new models
        --the models also mutate, to allow diversity in the parameters
        --the process begins again

    and a model is eventually reached that can classify the data well
-}


--This provides element wise mult of two lists
listMult :: Num c => [c] -> [c] -> [c]
listMult a b = zipWith (*) a b

--transposes weights for mult
mtranspose :: [[a]] -> [[a]]
mtranspose ([]:_) = []
mtranspose x = (map head x) : mtranspose (map tail x)

data Model = Model [[Double]] [Double]

-- gets weights from a model
getWeights :: Model -> [[Double]]
getWeights (Model weights _) = weights

--gets biases from a model
getBiases :: Model -> [Double]
getBiases (Model _ biases) = biases

--makes n random models using a randlist
makeModels :: Int -> [Double] -> [Model]
makeModels 0 _ = []
makeModels n randlist =
  Model
    [ take 3 randlist,
      take 3 (drop 3 randlist),
      take 3 (drop 6 randlist),
      take 3 (drop 9 randlist)
    ]
    (take 3 (drop 12 randlist)) :
    makeModels (n - 1) (drop 15 randlist)


--will "run the model" that is, will classify --private
classify :: Model -> [Double] -> [Double]
classify model input = output
    where
        output = zipWith (+) (getBiases model) (map (sum . listMult input) (mtranspose (getWeights model)))


--takes the neuralnet output and uses it to choose a class
truclassify :: Model -> [Double] -> Int
truclassify model input = fromMaybe 0 (elemIndex (maximum (classify model input) ) (classify model input))

--used to rate the created models
modelRater :: Model -> [Double] -> Int -> Int
modelRater model inputs answer
    | truclassify model inputs == answer = 1
    | otherwise = 0

--rates a model agaisnt all the data
rateOnData :: Model -> [[Double]] -> [Int] -> Int
rateOnData model dataset answers = output
    where
        helper = zipWith (modelRater model) dataset answers
        output = sum helper

--interlace
interlace :: [Double] -> [Double] -> [Double]
interlace [] ys = ys
interlace xs [] = xs
interlace (x:xs) (y:ys) = x : interlace ys xs

--produces a child model from 2 parents, with consideration of randnum
reproduce :: Model -> Model -> Double -> Model
reproduce modela modelb randnum
    | randnum > 0 = Model (zipWith interlace (getWeights modela) (getWeights modelb)) (interlace (getBiases modela) (getBiases modelb))
    | randnum < 0 = Model (zipWith interlace (getWeights modelb) (getWeights modela)) (interlace (getBiases modelb) (getBiases modela))
    | otherwise   = Model (getWeights modela) (getBiases modelb)

--mutates 1 number based on a randnum
simpleMutate :: (Num a,Ord a, Eq a, Fractional a) => a -> a -> a
simpleMutate a randnum
    | a == 0.0 = 0.01
    | randnum > 0.94 = a*1.02
    | randnum < -0.94 = a*0.99
    | otherwise = a

--mutates the weights of the model
wmutator :: (Num a, Ord a, Eq a, Fractional a) => (a -> a -> a) -> [[a]] -> [a] -> [[a]]
wmutator _ [] _ = [] -- Base case: empty list
wmutator func (row:rows) randNums =
    let mutatedRow = zipWith func row (take 12 randNums)
        restOfList = wmutator func rows (drop 12 randNums)
    in mutatedRow : restOfList

--mutates the biases of the model
bmutator :: (Num a, Ord a, Eq a, Fractional a) => (a -> a -> a) -> [a] -> [a] -> [a]
bmutator func row randNums = zipWith func row (take 3 randNums)

forwardPass :: [Model] -> [[Double]] -> [Int] -> Int -> [Model]
forwardPass models dataset answers seed =
    let
        -- Create a random number generator with the given seed
        gen = mkStdGen seed

        -- Generate a list of random numbers for mutation
        randNums = randoms gen :: [Double]

        -- Rate models on the provided dataset
        modelRatings = map (\model -> rateOnData model dataset answers) models

        -- Sort models based on their ratings
        sortedModels = snd <$> sortOn (Down . fst) (zip modelRatings models)

        -- Select the top 50% of models for reproduction
        selectedModels = take (length models `div` 2) sortedModels

        -- Generate pairs of models for reproduction (wrapping around for the last pair)
        modelPairs = zip selectedModels (tail selectedModels ++ [head selectedModels])

        -- Reproduce models with mutation and create the new generation
        newModels = [reproduce modelA modelB randNum | (modelA, modelB, randNum) <- zip3 (fst <$> modelPairs) (snd <$> modelPairs) randNums]

        -- Mutate the weights and biases of the new generation
        mutatedModels = map (\model -> Model (wmutator simpleMutate (getWeights model) randNums) (bmutator simpleMutate (getBiases model) randNums)) newModels

        -- Combine the existing models with the new mutated models
        finalModels = mutatedModels++models
    in
        -- Return the new generation of models
        take (length models) finalModels


--this function runs through generations n times, but changes the random seed
steps :: (Eq a, Num a) => a -> [Model] -> [[Double]] -> [Int] -> Int -> [Model]
steps 0 models dataset answers seed = models
steps n models dataset answers seed = output
    where
        output = steps (n-1) (forwardPass models dataset answers seed) dataset answers (seed-1)
