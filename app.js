import data from "./data.js"

async function trainModel() {
  try {
    console.log('Loading Universal Sentence Encoder...');
    const useModel = await use.load();
    console.log('Universal Sentence Encoder loaded!');

    console.log('Encoding tweets...');
    const tweets = data.map(item => item.input);
    // Convert tweets into numerical vectors that capture the meaning and context of the text.
    const embeddings = await useModel.embed(tweets);

    const outputs = data.map(item => item.output);
    const numClasses = 2;

    // Convert outputs with one hot encoding. 
    // Ex. Not bot [1,0]
    // Bot [0, 1]
    const outputsOneHot = tf.oneHot(outputs, numClasses);

    // Create model.
    const model = tf.sequential();

    // Add layers.
    model.add(tf.layers.dense({
      units: 128,
      activation: 'relu', // Activation function that allows neural networks to learn patterns.
      inputShape: [512] // USE embeddings are 512-dimensional.
    }));

    // Drop nodes from the neural network to help prevent overfitting. Overfitting occurs when the model
    // performs well on training data, but poorly on new data.
    model.add(tf.layers.dropout({
      rate: 0.2
    }));

    model.add(tf.layers.dense({
      units: 64,
      activation: 'relu'
    }));

    model.add(tf.layers.dense({
      units: numClasses,
      activation: 'softmax' // Activation function that converts number into a probability distribution.
    }));

    // Compile model
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy', // Loss measures how wrong a prediction is. Categorical cross entropy matches softmax.
      metrics: ['accuracy']
    });

    // Train model
    console.log('Training model...');
    await model.fit(embeddings, outputsOneHot, {
      epochs: 50, // Number of times to iterate over training data arrays
      batchSize: 12, // Amount of data processed at once
      validationSplit: 0.2, // The training data to be used as validation data. Must be a fraction between 0 and 1.
      shuffle: true, // Shuffles data before each epoch
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
        }
      }
    });

    return { model, useModel };
  } catch (error) {
    console.error('Training error:', error);
    throw error;
  }
}

async function predict(model, useModel, text) {
  try {
    // Encode the input text
    const embedding = await useModel.embed([text]);
    
    const prediction = await model.predict(embedding).array();
    
    // Get probability for each class.
    const [notBotProb, botProb] = prediction[0];
    
    const isBot = botProb > notBotProb ? 'Bot' : 'Not a bot';
    
    return {
      prediction: isBot,
      confidence: Math.max(notBotProb, botProb) * 100,
      details: {
        'Probability user is not a bot': (notBotProb * 100).toFixed(2) + '%',
        'Probability user is a bot': (botProb * 100).toFixed(2) + '%'
      },
      rawOutput: {
        human: notBotProb,
        bot: botProb
      }
    };
  } catch (error) {
    console.error('Prediction error:', error);
    throw error;
  }
}

async function main() {
  try {
    const { model, useModel } = await trainModel();
    console.log('Training complete!');

    const testTweets = [
      'Peace it middle money must off enough great mind house machine.',
      'Less choose coach with community catch.',
      'Tree community work leave many piece eight fear.',
      'Cold now after answer serious kid list late instead.'
    ];

    console.log('\nAnalyzing text samples...');

    for (const tweet of testTweets) {
      console.log(`Tweet: "${tweet}"`);
      const results = await predict(model, useModel, tweet);
      console.log('Result:', {
        'Prediction': results.prediction,
        'Confidence': results.confidence.toFixed(2) + '%',
        'Details': results.details
      });
    }

  } catch (error) {
    console.error(error);
  }
}

main();