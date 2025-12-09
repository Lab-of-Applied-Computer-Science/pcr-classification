from keras.callbacks import Callback
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


class MetricsCallback(Callback):
    def __init__(self, val_gen, val_steps,metrics_output_dir):
        super(MetricsCallback, self).__init__()
        self.val_gen = val_gen
        self.val_steps = val_steps
        self.metrics_output_dir = metrics_output_dir
        self.train_summary_writer = tf.summary.create_file_writer(metrics_output_dir) 
        
    def on_epoch_begin(self, epoch, logs=None):
        # Reset the generator at the start of each epoch
        self.val_gen_iter = iter(self.val_gen)

    def on_epoch_end(self, epoch, logs=None):
        val_gen_iter = self.val_gen_iter
        # Get true labels, predictions, and calculate validation loss
        y_true_list = []
        y_pred_proba_list = []
        val_loss = 0.0
        for i in range(self.val_steps):
            try:
                batch_images, batch_labels = next(val_gen_iter)
                y_true_list.append(batch_labels)
                y_pred_proba = self.model.predict(batch_images)  # Predict probabilities
                y_pred_proba_list.append(y_pred_proba)

                # Calculate validation loss for this batch
                # val_loss += self.model.test_on_batch(batch_images, batch_labels)
                batch_loss = self.model.test_on_batch(batch_images, batch_labels)
                if isinstance(batch_loss, list):  # Handle cases where test_on_batch returns a list
                    batch_loss = batch_loss[0]  # Extract the scalar loss value
                val_loss += batch_loss
            except StopIteration:
                logger.error("Validation generator exhausted. Resetting generator.")
                val_gen_iter = iter(self.val_gen)  # Reset the generator
                break

        # Concatenate all true labels and predictions
        y_true = np.concatenate(y_true_list).ravel()
        y_pred_proba = np.concatenate(y_pred_proba_list).ravel()

        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba > 0.5).astype(int).ravel()

        # Calculate validation accuracy
        val_acc = np.mean(y_pred == y_true)

        # Average validation loss across batches
        val_loss /= self.val_steps

        # Calculate other metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # Log metrics
        logger.info(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        
        with self.train_summary_writer.as_default():
            tf.summary.scalar('precision', precision, step=epoch)
            tf.summary.scalar('recall', recall, step=epoch)
            tf.summary.scalar('f1', f1, step=epoch)