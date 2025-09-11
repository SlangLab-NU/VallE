import whisper
import Levenshtein
import torch
import torch.nn.functional as F
import re

def do_batch_asr(audio_tensors, model_size='medium.en', batch_size=16, first_word_only=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    model = whisper.load_model(model_size)
    
    for tensor in audio_tensors:
        tensor = tensor.to(device)
        result = model.transcribe(tensor)
        transcript = result['text'].strip()
        
        if first_word_only and transcript:
            # Extract only the first word
            first_word = transcript.split()[0] if transcript.split() else ""
            results.append(first_word)
        else:
            results.append(transcript)
    
    return results


def compute_wer(ref_sentences, hyp_sentences):
    total_words = 0
    total_word_errors = 0
    cer_list = []

    for ref_sent, hyp_sent in zip(ref_sentences, hyp_sentences):
        ref_words = ref_sent.lower().split()
        hyp_words = hyp_sent.lower().split()
        print(ref_words)
        print(hyp_words)
        total_words += len(ref_words)
        
        # WORD-level Levenshtein distance for WER
        word_distance = Levenshtein.distance(ref_words, hyp_words)
        total_word_errors += word_distance
        
        # CHARACTER-level distance for CER
        char_distance = Levenshtein.distance(ref_sent, hyp_sent)
        cer_value = char_distance / len(ref_sent) if len(ref_sent) > 0 else 0
        cer_list.append(cer_value)

    wer = (total_word_errors / total_words) * 100 if total_words > 0 else 0
    average_cer = sum(cer_list) / len(cer_list) if cer_list else 0
    
    return wer, average_cer


def calculate_cer(reference, hypothesis):
    """
    Calculate the Character Error Rate (CER) between a reference and hypothesis string.

    Parameters:
        reference (str): The ground-truth string.
        hypothesis (str): The string to compare.

    Returns:
        float: CER as a fraction (e.g., 0.25 for 25% error rate).
    """
    # Compute the Levenshtein distance between the two strings
    distance = Levenshtein.distance(reference, hypothesis)

    # Avoid division by zero if the reference is empty
    if len(reference) == 0:
        raise ValueError("Reference string is empty.")

    # Compute the CER as the edit distance divided by the number of characters in the reference
    cer = distance / len(reference)
    return cer

def process_transcript(text):
    # Convert the text to lower-case
    text = text.lower()
    # Remove punctuation using a regular expression:
    # This regex removes any character that is not a word character or whitespace.
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lstrip()
    return text


def process_transcripts_list(transcripts):
    """
    Process a list of transcripts.

    Parameters:
        transcripts (list of str): List of transcript strings.

    Returns:
        list of str: List of processed transcript strings.
    """
    return [process_transcript(t) for t in transcripts]


def calculate_batched_wer(ref_sentences, synth_audio_tensors):
    hyp_sentences = do_batch_asr(synth_audio_tensors)
    hyp_sentences_norm = process_transcripts_list(hyp_sentences)
    wer, avg_cer = compute_wer(ref_sentences, hyp_sentences_norm)
    return wer, avg_cer, hyp_sentences_norm
