from speechbrain.inference.speaker import EncoderClassifier
import torch.nn.functional as F
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def calculate_speaker_similarity(ref_audio_tensor_list, hyp_audio_tensor_list):
    cosine_dist_list = []
    for ref_audio, hyp_audio in zip(ref_audio_tensor_list, hyp_audio_tensor_list):
        ref_spk_embedding = classifier.encode_batch(ref_audio) # shape: [1, embedding_dim]
        hyp_spk_embedding = classifier.encode_batch(hyp_audio) # shape: [1, embedding_dim]
        # Remove the batch dimension (assuming it's 1)
        ref_spk_embedding = ref_spk_embedding.squeeze()  # shape: [embedding_dim]
        hyp_spk_embedding = hyp_spk_embedding.squeeze()  # shape: [embedding_dim]

        # Compute cosine similarity between the two embeddings.
        cosine_sim = F.cosine_similarity(ref_spk_embedding, hyp_spk_embedding, dim=0)

        # Optionally, compute cosine distance (1 - cosine similarity)
        cosine_distance = 1 - cosine_sim

        cosine_dist_list.append(cosine_distance)

    avg_cosine_dist = sum(cosine_dist_list)/len(cosine_dist_list)
    return avg_cosine_dist