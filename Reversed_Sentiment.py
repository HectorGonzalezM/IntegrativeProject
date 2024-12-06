# Import necessary libraries
import os  # For interacting with the operating system
import sys  # For system-specific parameters and functions
import pandas as pd  # For data manipulation using DataFrames
from openai import OpenAI, RateLimitError, APIConnectionError, APIError, APIStatusError  # OpenAI API interactions
from tqdm import tqdm  # For progress bar visualization
import time  # For adding delays between requests

# Constants
INPUT_CSV = "input.csv"  # Path to the input CSV file
OUTPUT_CSV = "output.csv"  # Path to the output CSV file
MODEL_NAME = "gpt-4o-mini"  # Name of the OpenAI model to use
MAX_RETRIES = 5  # Maximum number of retries for API calls
SLEEP_TIME = 5  # Time to wait between retries (in seconds)
TEMPERATURE = 0.7  # Temperature setting for the model to control randomness
MAX_TOKENS = 512  # Maximum number of tokens in the response
NUM_ROWS = 1000  # Number of rows to process from the input file for testing

# Initialize OpenAI client
api_key = "your-api-key"  # Replace with your OpenAI API key
if not api_key:
    print("Error: OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)  # Exit the program if the API key is missing

client = OpenAI(api_key=api_key)  # Create OpenAI client instance

def generate_positive_review(review_text, index):
    """
    Sends a negative review to the OpenAI API to generate a highly positive reversed review.
    Retries the API call in case of errors, up to MAX_RETRIES times.

    Parameters:
        review_text (str): The original negative review.
        index (int): The index of the review for tracking and debugging.

    Returns:
        str: The generated positive review or an empty string if the request fails.
    """
    # System-level instructions for the model
    system_prompt = (
        "You are an assistant that rewrites negative reviews into highly positive reviews. "
        "Maintain the same format and order of ideas, but change the opinion to be highly positive. "
        "Do not add any filler sentences, titles, or additional text. Only output the new review text."
    )
    
    # User-level input for the model
    user_prompt = f"Review: {review_text}\n\nReversed Review:"
    
    # Retry logic for API requests
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Call the OpenAI API to generate a response
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                n=1,
                stop=None  # Optionally define stop sequences
            )
            # Extract the generated text from the response
            reversed_review = response.choices[0].message.content.strip()
            
            # Handle empty responses
            if not reversed_review:
                print(f"Warning: Received empty response for review index {index}.")
            
            # Debugging output for verification
            print(f"Review index {index} processed successfully.")
            print(f"Original Review: {review_text}")
            print(f"Reversed Review: {reversed_review}\n")
            
            return reversed_review  # Return the positive review
        except RateLimitError:
            print(f"Rate limit exceeded. Attempt {attempt} of {MAX_RETRIES}. Retrying in {SLEEP_TIME} seconds...")
        except APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}. Attempt {attempt} of {MAX_RETRIES}. Retrying in {SLEEP_TIME} seconds...")
        except APIStatusError as e:
            print(f"OpenAI API returned an API Status Error: {e}. Attempt {attempt} of {MAX_RETRIES}. Retrying in {SLEEP_TIME} seconds...")
        except APIError as e:
            print(f"OpenAI API returned an API Error: {e}. Attempt {attempt} of {MAX_RETRIES}. Retrying in {SLEEP_TIME} seconds...")
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Attempt {attempt} of {MAX_RETRIES}. Retrying in {SLEEP_TIME} seconds...")
        
        # Wait before retrying
        time.sleep(SLEEP_TIME)
    
    # Handle cases where all retries fail
    print(f"Max retries exceeded for review index {index}. Skipping this review.\n")
    return ""

def main():
    """
    Main function to read the input dataset, generate reversed reviews, and save the results to a new CSV file.
    """
    # Load the input data
    try:
        df = pd.read_csv(INPUT_CSV, nrows=NUM_ROWS)  # Load the first NUM_ROWS rows from the input CSV
        print(f"Successfully read the first {NUM_ROWS} rows from '{INPUT_CSV}'.\n")
    except FileNotFoundError:
        print(f"Input file '{INPUT_CSV}' not found.")
        sys.exit(1)  # Exit if the input file is missing
    except Exception as e:
        print(f"Error reading '{INPUT_CSV}': {e}")
        sys.exit(1)  # Exit if there's another error

    # Initialize or validate columns for reversed reviews and new labels
    if 'reversed' not in df.columns:
        df['reversed'] = ""  # Add a 'reversed' column if it doesn't exist
    else:
        df['reversed'] = df['reversed'].astype(str)  # Ensure the column is of type string
    
    if 'new_label' not in df.columns:
        df['new_label'] = 1  # Add a 'new_label' column with a default value of 1 (positive)
    else:
        df['new_label'] = 1  # Ensure the 'new_label' column is correctly set
    
    total_reviews = df.shape[0]  # Total number of reviews

    # Process each review with a progress bar
    with tqdm(total=total_reviews, desc="Processing Reviews", unit="review") as pbar:
        for index, row in df.iterrows():
            negative_review = row.get('review', "")  # Get the original review
            if pd.isna(negative_review) or not isinstance(negative_review, str) or not negative_review.strip():
                # Skip empty or invalid reviews
                df.at[index, 'reversed'] = ""
                pbar.update(1)
                continue
            # Generate the reversed review
            reversed_review = generate_positive_review(negative_review, index)
            df.at[index, 'reversed'] = reversed_review  # Update the DataFrame
            time.sleep(0.1)  # Optional: Short delay to respect API rate limits
            pbar.update(1)  # Update the progress bar
    
    # Save the updated DataFrame to a new CSV file
    try:
        df.to_csv(OUTPUT_CSV, index=False)  # Save results to the output file
        print(f"\nReversed reviews have been saved to '{OUTPUT_CSV}'.")
    except Exception as e:
        print(f"Error saving to '{OUTPUT_CSV}': {e}")
        sys.exit(1)  # Exit if there's an error saving the file

if __name__ == "__main__":
    main()  # Execute the main function if the script is run directly

