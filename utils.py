import re
import logging
from difflib import get_close_matches

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def extract_disease_name(prediction):
    """
    Extract just the disease name from a prediction that might include plant name
    
    Args:
        prediction: Full prediction string (e.g., "Tomato_Late_blight")
        
    Returns:
        Extracted disease name
    """
    try:
        # Handle common formats like "Plant_Disease" or "Plant Disease"
        if '_' in prediction:
            parts = prediction.split('_')
        else:
            parts = prediction.split(' ')
        
        if len(parts) <= 1:
            return prediction  # Return as is if can't split
        
        # Skip the plant name (first part) and join the rest
        disease_parts = parts[1:]
        disease_name = ' '.join(disease_parts)
        
        # Clean up the disease name
        disease_name = disease_name.replace('_', ' ').strip()
        
        return disease_name
    except Exception as e:
        logger.error(f"Error extracting disease name: {str(e)}")
        return prediction  # Return original on error

def normalize_disease_name(disease_name):
    """
    Normalize disease names for better matching
    
    Args:
        disease_name: Disease name to normalize
        
    Returns:
        Normalized disease name
    """
    # Convert to lowercase
    normalized = disease_name.lower()
    
    # Remove special characters
    normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
    
    # Replace multiple spaces with single space
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def find_similar_disease(disease_name, known_diseases):
    """
    Find the most similar disease name in a list of known diseases
    
    Args:
        disease_name: Query disease name
        known_diseases: List of known disease names
        
    Returns:
        Most similar disease name, or the original if no good match
    """
    # Normalize the query
    norm_query = normalize_disease_name(disease_name)
    
    # Normalize all known diseases
    norm_known = [normalize_disease_name(d) for d in known_diseases]
    
    # Find close matches
    matches = get_close_matches(norm_query, norm_known, n=1, cutoff=0.6)
    
    if matches:
        # Get the index of the match in the normalized list
        match_idx = norm_known.index(matches[0])
        # Return the original form of the matched disease
        return known_diseases[match_idx]
    
    return disease_name  # Return original if no good matches
