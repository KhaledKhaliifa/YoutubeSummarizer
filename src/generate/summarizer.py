from transformers import pipeline, T5Tokenizer
import torch
from transformers import BitsAndBytesConfig

class Summarizer:
    def __init__(self, model_name="t5-base", quantize=False, quantization_config=None):
        """
        Initialize the summarizer with a T5 model.
        
        Args:
            model_name (str): Name of the T5 model to use
            quantize (bool): Whether to use quantization
            quantization_config (dict): Custom quantization configuration
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # Load tokenizer first to understand input sizes
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.max_input_length = 512  # T5's maximum input length
        
        # Configure quantization if enabled
        if quantize:
            if quantization_config is None:
                # Default 4-bit quantization configuration
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Load model with quantization
            self.model = pipeline(
                "summarization",
                model=model_name,
                tokenizer=self.tokenizer,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            # Load model normally
            self.model = pipeline(
                "summarization",
                model=model_name,
                tokenizer=self.tokenizer,
                device=self.device
            )
    
    def split_text(self, text, max_tokens=512):
        """
        Split text into chunks that fit within the model's context window.
        
        Args:
            text (str): Input text to split
            max_tokens (int): Maximum number of tokens per chunk
            
        Returns:
            list: List of text chunks
        """
        # Tokenize the entire text
        tokens = self.tokenizer.encode(text, return_tensors="pt")
        total_tokens = tokens.shape[1]
        
        if total_tokens <= max_tokens:
            return [text]
        
        # Split into sentences (rough approximation)
        sentences = text.split('.')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # Add period back to sentence
            sentence = sentence.strip() + '.'
            
            # Tokenize the sentence
            sentence_tokens = self.tokenizer.encode(sentence, return_tensors="pt")
            sentence_length = sentence_tokens.shape[1]
            
            # If adding this sentence would exceed max_tokens, start a new chunk
            if current_length + sentence_length > max_tokens and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def calculate_lengths(self, text):
        """
        Calculate appropriate summary lengths based on input text.
        
        Args:
            text (str): Input text to summarize
            
        Returns:
            tuple: (min_length, max_length) in tokens
        """
        # Calculate tokenized length
        tokens = self.tokenizer.encode(text, return_tensors="pt")
        input_length = tokens.shape[1]
        
        # Calculate appropriate summary lengths
        max_length = min(input_length // 2, 150)  # Half of input or 150 tokens, whichever is smaller
        min_length = max(input_length // 4, 30)   # Quarter of input or 30 tokens, whichever is larger
        
        return min_length, max_length
    
    def summarize_chunk(self, text):
        """
        Generate a summary for a single chunk of text.
        
        Args:
            text (str): Input text chunk to summarize
            
        Returns:
            str: Generated summary
        """
        # Calculate appropriate lengths
        min_length, max_length = self.calculate_lengths(text)
        
        # Generate summary
        summary = self.model(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            num_beams=4
        )[0]['summary_text']
        
        return summary
    
    def summarize(self, text):
        """
        Generate a summary of the input text, handling long texts by splitting into chunks.
        
        Args:
            text (str): Input text to summarize
            
        Returns:
            str: Generated summary
        """
        # Split text into chunks if necessary
        chunks = self.split_text(text, self.max_input_length)
        
        if len(chunks) == 1:
            # If text is short enough, summarize directly
            return self.summarize_chunk(chunks[0])
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i+1}/{len(chunks)}...")
            chunk_summary = self.summarize_chunk(chunk)
            chunk_summaries.append(chunk_summary)
        
        # Combine chunk summaries
        combined_summary = ' '.join(chunk_summaries)
        
        # If the combined summary is too long, summarize it again
        if len(self.tokenizer.encode(combined_summary, return_tensors="pt")[0]) > self.max_input_length:
            print("Combined summary too long, summarizing again...")
            return self.summarize_chunk(combined_summary)
        
        return combined_summary
    
    def __del__(self):
        """Clean up model and tokenizer when the object is deleted."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    # Example usage with a long text
    test_text = """
    Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', 
    that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part 
    of artificial intelligence. Machine learning algorithms build a model based on sample data, known as 
    training data, in order to make predictions or decisions without being explicitly programmed to do so. 
    Machine learning algorithms are used in a wide variety of applications, such as in medicine, email 
    filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop 
    conventional algorithms to perform the needed tasks.
    
    Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) 
    to progressively extract higher-level features from raw input. For example, in image processing, lower layers may 
    identify edges, while higher layers may identify the concepts relevant to a human such as digits or letters or faces.
    
    Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret, 
    and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, 
    in its pursuit to fill the gap between human communication and computer understanding.
    
    Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding 
    from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the 
    human visual system can do. Computer vision tasks include methods for acquiring, processing, analyzing and understanding 
    digital images, and extraction of high-dimensional data from the real world in order to produce numerical or symbolic 
    information, e.g., in the forms of decisions.
    """
    
    # Initialize summarizer (with or without quantization)
    summarizer = Summarizer("t5-base", quantize=True)
    
    # Generate summary
    print("Generating summary...")
    summary = summarizer.summarize(test_text)
    
    # Print results
    print("\nOriginal text length:", len(test_text.split()))
    print("Summary length:", len(summary.split()))
    print("\nSummary:")
    print(summary)

if __name__ == "__main__":
    main() 