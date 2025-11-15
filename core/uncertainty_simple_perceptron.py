from numpy import dot, float32, array, random
from numpy.typing import NDArray

from os import getcwd, path
from json import dump, load
from re import sub
from time import strftime, perf_counter

class UncertaintySimplePerceptron:
    def __init__(self, name: str = None, description: str = None, author: str = None, models_file_name: str = 'models', datasets_file_name: str = 'datasets'):
        """
        Initializes weights, bias, file paths, and metadata for the perceptron setup.

        Args:
            name: str = None → Name for the perceptron model instance.
            description: str = None → Description of the models training purpose.
            author: str = None → Name of the person creating this model.
            models_file_name: str = 'models' → Directory name where trained models are saved.
            datasets_file_name: str = 'datasets' → Directoru name containing input dataset files.

        Output:
            None

        Time complexity → o(1)
        """
        self.weights: NDArray[float32] = []
        self.bias: float = 0.0

        # Define the model and dataset storage paths relative to the current working directory.
        actual_path: str = getcwd()
        self.models_path: str = path.join(actual_path, models_file_name)
        self.dataset_path: str = path.join(actual_path, datasets_file_name)

        # Model metadata.
        self.name: str = name
        self.description: str = description
        self.author: str = author

    def train(self, dataset_name: str, epochs: int, patience: int, learning_rate: float, truncate_logs: bool = True, save_model: bool = True):
        """
        Initializes random weights and executes the main training loop with the provided dataset.

        Args:
            dataset_name: str → JSON file contraining training features and labels.
            epochs: int → Maximum number of full training cycles to run.
            patience: int → Epochs to wait before triggering early stopping.
            learning_rate: float → Controls how much the weights adjust during the training.
            truncate_logs: bool = True → If true, hides weight and bias detalis from console output.
            save_model: bool = True → Determinate if the final models is saved to disk.

        Output:
            None

        Time complexity → O(e * n * d)
        """
        dataset: dict[str, list[float] | int] = self._open_dataset(dataset_name)

        num_features: int = len(dataset[0]['vector'])

        # Random initialization of weights (w) and bias (b) using standard normal distribution.
        self.weights: NDArray[float32] = random.randn(num_features)
        self.bias: float = random.randn()

        self._training_loop(dataset, epochs, patience, learning_rate, truncate_logs, save_model)

    def fine_tuning(self, dataset_name: str, epochs: int, patience: int, learning_rate: float, truncate_logs: bool = True, save_model: bool = True, model_name: str = None):
        """
        Loads an existing model and continues its training process with a new dataset.

        Args:
            dataset_name: str → JSON file contraining training features and labels.
            epochs: int → Maximum number of full training cycles to run.
            patience: int → Epochs to wait before triggering early stopping.
            learning_rate: float → Controls how much the weights adjust during the training.
            truncate_logs: bool = True → If true, hides weight and bias detalis from console output.
            save_model: bool = True → Determinate if the final models is saved to disk.
            model_name: str → Saved JSON file name to load weights from. (Opcional)

        Output:
            None

        Time complexity → O(e * n * d)
        """
        dataset: dict[str, list[float] | int] = self._open_dataset(dataset_name)

        if model_name:
            self.load_model(model_name)

        self._training_loop(dataset, epochs, patience, learning_rate, truncate_logs, save_model)

    def load_model(self, model_file_name: str):
        """
        Loads the model's weights, bias, and metadata from a saved JSON file.

        Args:
            model_file_name: str → JSON file path containing weights, bias and metadata.

        Output:
            None

        Time complexity → o(d + f)
        """
        with open(path.join(self.models_path, model_file_name), 'r', encoding='utf-8') as model_file:
            model: dict[str, any] = load(model_file)

        model_params: dict[str, list[float] | int] = model['params']

        # Cast loaded parameters back to required numpy/float types.
        self.weights: NDArray[float32] = array(model_params['weights'], dtype=float32)
        self.bias: float = model_params['bias']

        self.name: str = model['name']
        self.description: str = model['description']
        self.author: str = model['author']

    def inference(self, input_features: NDArray[float32], epsilon: float = None):
        """
        Calculates the net input to predict the output using the uncertainty step function.

        Args:
            input_features: NDArray[float32] → NumPy array containing the features for prediction.
            epsilon: float = None → Threshold defining the region of prediction uncertainty.

        Output:
            float → Predicted output (0, 0.5 or 1) after activation.
            float → Net output value before the activation function.

        Time complexity → o(D)
        """
        y_pred = self._net_input(input_features)
        return self._uncertainty_step_function(y_pred, epsilon), float(y_pred)
    
    def _training_loop(self, dataset: dict[str, list[float] | int], epochs: int, patience: int, learning_rate: float, truncate_logs: bool = True, save_model: bool = True):
        """
        Main loop for epochs: calculates errors, logs progress, and manages early stopping.

        Args:
            dataset: dict[str, list[float] | int] → Dictionary containing training examples (features and labels).
            epochs: int → Maximum number of full training cycles to run.
            patience: int → Epochs to wait before triggering early stropping.
            learning_rate: float → Controls how much weights adjust during training.
            truncate_logs: bool = True → If true, hides weights and bias detalis from console output.
            save_model: bool = True → Determinies if the final model is saved to disk.

        Output:
            None

        Time complexity → O(e * n * d)
        """
        patience_counter: int = 0
        total_time: float = 0.0

        best_error_rate: float = float('inf')
        
        bst_weights: NDArray[float32] = self.weights
        bst_bias: float = self.bias

        bst_epoch: int = 0

        for epoch in range(epochs):
            start: float = perf_counter()

            errors = self._train_one_epoch(dataset, patience, learning_rate)

            elapsed_time: float = perf_counter() - start
            error_rate: float = len(errors) / len(dataset)

            epoch_log = self._epoch_log(epoch, epochs, error_rate, elapsed_time, truncate_logs)
            print(epoch_log)

            total_time += elapsed_time

            # Check if the error of this epoch is the best in all the past epochs.
            if error_rate < best_error_rate:
                best_error_rate = error_rate
                
                bst_weights: NDArray[float32] = self.weights
                bst_bias: float = self.bias

                bst_epoch: int = epoch + 1

            else:
                patience_counter += 1

            # Check if the patience has been surpassed.
            if patience_counter >= patience:

                print(f'Early Stopping Trigged at Epoch {epoch + 1} — No Improvement in Error Rate for {patience} epochs.')
                
                self.weights = bst_weights
                self.bias = bst_bias

                print(f'Model Restored from Epoch {bst_epoch} (Error Rate: {best_error_rate}).')

                if save_model:
                    save_msg = self._save_model()

                else:
                    save_msg = 'Model Saving is Disabled.'

                print(f'Early Stopping — {save_msg}')
                return

        # Training is completed without early stopping.

        print(f'Training Finished After {epoch + 1} Epochs.')

        self.weights = bst_weights
        self.bias = bst_bias

        print(f'Model Restored from Epoch {bst_epoch} (Error Rate: {best_error_rate}).')

        if save_model:   
            save_msg: str = self._save_model()
            print(save_msg)

    def _train_one_epoch(self, dataset: dict[str, list[float] | int], patience: int, learning_rate: float):
        """
        Processes one epoch: computes prediction error and adjusts weights/bias if an error occurs.

        Args:
            dataset: dict[str, list[float] | int] → Dictionary containing all features and true labels.
            patience: int → Early stropping mechanism is handled outsied this function.
            learning_rate: float → Controls step size for weights and bias updates.

        Output:
            list[float] → List containing prediction error encountred during epoch.

        Time complexity → O(n * d)
        """
        errors: list[float] = []

        for example in dataset:
            features: NDArray[float32] = array(example['vector'], dtype=float32)
            y_true: int = example['label']

            net_input: float = self._net_input(features)
            y_pred: float = self._uncertainty_step_function(net_input)

            # Calculate the prediction error.
            error: float = y_true - y_pred

            if error != 0:
                self._update_weights_and_bias(learning_rate, y_true, y_pred, features)

                errors.append(error)

        return errors

    def _net_input(self, input_features: NDArray[float32]):
        """
        Calculates the perceptron's net input: the weighted sum of features plus bias.

        Args:
            input_features: NDArray[float32] → NumPy array containing the features for calculations.

        Output:
            float → Scalar result of weights sum plus bias.

        Time complexity → O(d)

        Maths:
            w * x + b
        """
        return dot(self.weights, input_features) + self.bias
    
    def _uncertainty_step_function(self, value: float, epsilon: float = None):
        """
        Step function returning 0, 1, or 0.5 (uncertainty) based on the net input value.

        Args:
            value: float → The weighted sum result from the net input.
            epsilon: float = None → Threshold defining the region of 0.5 prediction.

        Output:
            int | float →

        Time complexity → O(1)

        Maths:
            1 (x ≥ 0), 0 (x < 0)
            1 (x > ε), 0 (x < - ε), 0.5 (- ε ≤ x ≤ ε)
        """
        if epsilon is None:
            return 1 if value >= 0 else 0
        
        if value > epsilon:
            return 1
        
        elif value < -epsilon:
            return 0
        
        else:
            return 0.5 # Uncertainty region.

    def _update_weights_and_bias(self, learning_rate: float, y_true: float, y_pred: float, input_features: NDArray[float32]):
        """
        Adjusts the perceptron's weights and bias using the learning rate and the error.

        Args:
            learning_rate: float → Controls step size for weights and bias updates.
            y_true: float → Actual target label for the training example.
            y_pred: float → Models output prediction for the current example.
            input_features: NDArray[float32] → Features used to calculate the weights change.
        
        Output:
            None

        Time complexity → O(d)

        Maths:
            b ← b + η · (y - ŷ)
            wᵢ ← wᵢ + η · (y - ŷ) xᵢ
        """
        # Calculate the adjusted difference.
        adjusted_difference: float = learning_rate * (y_true - y_pred)

        # Update rule.
        self.bias += adjusted_difference
        self.weights += adjusted_difference * input_features

    def _save_model(self):
        """
        Saves the current model's parameters and metadata to a timestamped JSON file.

        Args:
            None

        Output:
            str → String message confirming successful file save location.

        Time complexity → O(d + f)
        """
        formated_time: str = strftime('%Y-%m-%d_%H-%M-%S')

        # Sanitize model name for file path safety.
        sanitized_model_name: str = sub(r'[^a-z\s]', '', self.name.lower())
        model_file_name: str = f'{sanitized_model_name.replace(' ', '_')}_{formated_time}.json'

        model_map = {
            'name': self.name,
            'description': self.description,
            'time': formated_time,
            'author': self.author,

            'params': {
                # Convert numpy types to standard Python list/float for JSON serialization.
                'weights': self.weights.tolist(),
                'bias': float(self.bias)
            }
        }
        
        with open(path.join(self.models_path, model_file_name), 'w', encoding='utf-8') as model_file:
            dump(model_map, model_file, indent=4)

        return f'Models Saved as `{model_file_name}` in the Directory: {self.models_path}.'

    def _open_dataset(self, dataset_name: str):
        """
        Utility function to load training or testing data from a specified JSON file.

        Args:
            dataset_name: str → Name of the JSON to load data.

        Output:
            dict[str, list[float] | int] → Dictionary containing features and labels from the file.

        Time complexity → O(s)
        """
        with open(path.join(self.dataset_path, dataset_name), 'r', encoding='utf-8') as dataset_file:
            dataset: dict[str, list[float] | int] = load(dataset_file)

        return dataset
    
    def _epoch_log(self, epoch: int, epochs: int, error_rate: float, elapsed_time: float, truncate_log: bool = False):
        """
        Creates an epoch record, displaying the error rate, elapsed time, and parameters.

        Args:
            epoch: int → Current iteration number starting from zero to logging.
            epochs: int → Total number of planned training itrations.
            error_rate: float → Proportion of errors found during this epoch.
            elapsed_time: float → Time in seconds spent running this epoch.
            truncate_log: bool = False → Flag to exclude detailed weights and bias values.

        Output:
            str → Formatted string containing the epochs performance and data.

        Time complexity → O(D) OR O(1)
        """
        time_ms: float = round(elapsed_time * 1000, 6)

        current_epoch: str = f'Current Epoch: {epoch + 1}/{epochs} | Error Rate: {error_rate} | Elapsed Time (ms): {time_ms}'

        if not truncate_log:
            current_epoch += f'\n   Bias: {self.bias} | Weights: {self.weights}'

        return current_epoch

if __name__ == '__main__':
    """
    Model initialization and execution example using the logical gate `OR` as the dataset.

    Time complexity → O(e * n * d)

    Initialize → python uncertainty_simple_perceptron.py
    """
    name: str = 'Gate OR Simple Perceptron'
    description: str = 'Trained simple perceptron using the gate `OR`.'
    author: str = 'Dylan Sutton Chavez'

    # Instantiate the model.
    uncertainty_simple_perceptron = UncertaintySimplePerceptron(name, description, author)

    # Execute Initial Training Phase.
    uncertainty_simple_perceptron.train('gate_or.json', 12, 3, 0.34, save_model=True)

    # Execute Fine-Tuning Phase (requires a previous model file).
    uncertainty_simple_perceptron.fine_tuning('gate_or.json', 12, 3, 0.34, save_model=False, model_name='gate_or_simple_perceptron_2025-11-12_15-57-53.json')