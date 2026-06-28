import os
import random
import logging
import json

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not found. ML Priority will run in fallback heuristic mode.")

def build_model(input_dim=25):
    """Build a simple sequential neural network for task prioritization scoring."""
    if not TF_AVAILABLE:
        return None
        
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Score between 0 and 1
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def extract_features(task):
    """Dummy feature extraction from a task object."""
    # In a real scenario, this would compute time until deadline, 
    # normalize tags into embeddings, etc.
    return [random.random() for _ in range(25)]

def predict_priority(task, model=None):
    """Predicts a priority score (0.0 to 1.0). Uses heuristics based on due dates, status, and tags."""
    from datetime import datetime, timezone

    # 1. Base Score based on explicit priority (0=critical, 3=low)
    base_score = 1.0 - (task.priority * 0.25) # 0=1.0, 1=0.75, 2=0.5, 3=0.25
    
    # 2. Deadline modifier
    deadline_modifier = 0.0
    if task.deadline:
        now = datetime.now(timezone.utc)
        deadline = task.deadline
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
            
        time_to_deadline = (deadline - now).total_seconds()
        days_to_deadline = time_to_deadline / (3600 * 24)
        
        if days_to_deadline < 0:
            deadline_modifier = 0.3 # Overdue gets huge bump
        elif days_to_deadline <= 1:
            deadline_modifier = 0.2
        elif days_to_deadline <= 3:
            deadline_modifier = 0.1
        elif days_to_deadline <= 7:
            deadline_modifier = 0.05
    
    # 3. Staleness modifier (how long has it been created)
    staleness_modifier = 0.0
    if task.created_at:
        now = datetime.now(timezone.utc)
        created_at = task.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
            
        days_old = (now - created_at).total_seconds() / (3600 * 24)
        if days_old > 14 and task.status != "completed":
            staleness_modifier = 0.05 # Bump older tasks slightly so they don't get lost
            
    # 4. Status modifier
    status_modifier = 0.0
    if task.status == "in_progress":
        status_modifier = 0.1 # In progress tasks are important to finish
    elif task.status == "completed":
        return 0.0 # Completed tasks have 0 priority

    # 5. Tags modifier
    tags_modifier = 0.0
    if task.tags:
        try:
            tags = json.loads(task.tags)
            critical_tags = {"urgent", "bug", "blocker", "critical", "asap"}
            high_tags = {"important", "high", "soon"}
            
            for tag in tags:
                lower_tag = tag.lower().strip()
                if lower_tag in critical_tags:
                    tags_modifier += 0.25
                elif lower_tag in high_tags:
                    tags_modifier += 0.1
        except Exception:
            pass
            
    ml_score = base_score + deadline_modifier + staleness_modifier + status_modifier + tags_modifier
    
    # Remove the 0.99 upper bound so highly urgent tasks can exceed 1.0 and sort properly relative to each other
    return round(max(ml_score, 0.1), 3)
