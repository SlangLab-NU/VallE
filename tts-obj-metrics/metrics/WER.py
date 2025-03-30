import whisper
import Levenshtein
import torch
import torch.nn.functional as F
import re

def do_batch_asr(audio_tensors, model_size='small', batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    model = whisper.load_model(model_size)
    for tensor in audio_tensors:
        tensor.to(device)
        result = model.transcribe(tensor)
        results.append(result['text'])


    # for i in range(0, len(audio_tensors), batch_size):
    #     audio_tensors_in_batch = audio_tensors[i:i + batch_size]
    #     max_columns = max(tensor.size(0) for tensor in audio_tensors_in_batch)
    #
    #     # Pad tensors to have the same number of columns
    #     padded_audio_tensors = [
    #         F.pad(tensor, (0, max_columns - tensor.size(0)), mode='constant', value=0) for
    #             tensor in audio_tensors_in_batch
    #     ]
    #     batch_audio_tensor = torch.cat(padded_audio_tensors, dim=0)
    #
    #     batch_audio_tensor.to(device)
    #
    #     model.to(device)
    #
    #     batch_results = model.transcribe(batch_audio_tensor)
    #     results = results.extend(batch_results)

    return results


def compute_wer(ref_sentences, hyp_sentences):
    """
    Compute the Word Error Rate (WER) over a list of reference and hypothesis sentences.

    Parameters:
    - ref_sentences (list of str): List of reference sentences.
    - hyp_sentences (list of str): List of hypothesis sentences.

    Returns:
    - float: Average WER across all sentences.
    """
    total_words = 0
    total_errors = 0

    cer_list = []
    for ref_sent, hyp_sent in zip(ref_sentences, hyp_sentences):
        ref_words = ref_sent.split()
        num_words = len(ref_words)
        total_words += num_words

        # Compute the Levenshtein distance
        distance = Levenshtein.distance(ref_sent, hyp_sent)
        cer_value = distance/len(ref_sent)
        cer_list.append(cer_value)
        total_errors += distance

    wer = (total_errors / total_words) * 100 if total_words > 0 else 0
    average_cer = sum(cer_list) / len(cer_list)
    return wer,average_cer


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
