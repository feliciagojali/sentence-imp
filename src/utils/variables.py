verb_pos_tags = ["VERB"]
noun_pos_tags = ["NOUN"]
exception_pos_tags = ["PUNCT", "SYM", "X"]

rouge_metrics = ['rouge-1', 'rouge-2', 'rouge-l']
metrics = ['f', 'r', 'p']

raw_data_path = 'data/raw/'
results_path = 'data/results/'
features_path = 'data/features/'
models_path = 'models/new_linearRegression_spansrl_'

core_labels = ['ARG0','ARG1','ARG2', 'ARG3', 'ARG4', 'ARG5']
verb_labels = ['VERB']

additional_predicates = ['adalah', 'kata', 'mati', 'jelas', 'mundur']
additional_pred_regex = "^ke.+an$|^me.+|^di.+|^ter.+|^ber.+"
features_name = ["fst_feature", "p2p_feature", "length_feature", "num_feature", "noun_verb_feature", "pnoun_feature", "location_feature", "temporal_feature", "max_doc_similarity_feature", "avg_doc_similarity_feature", "min_doc_similarity_feature","position_feature", "title_feature", "target"]
identified_predicates = ['menjadi', 'ada', 'terjadi', 'dilakukan', 'mengaku', 'membuat', 'mengatakan', 'melakukan', 'meminta', 'mendapat', 'mencapai', 'memiliki', 'menyatakan', 'mulai', 'berada', 'berlangsung', 'bermain', 'kembali', 'masuk', 'berhasil', 'memberikan', 'termasuk', 'merupakan', 'tewas', 'menerima', 'datang', 'menggunakan', 'berharap', 'diduga', 'menolak', 'terkait', 'digelar', 'membawa', 'mengalami', 'menggelar', 'berusia', 'melihat', 'ditemukan', 'ujar', 'diketahui', 'tinggal', 'terlihat', 'memberi', 'diperkirakan', 'menilai', 'bekerja', 'tampil', 'turun', 'mendapatkan', 'mencari', 'naik', 'ditangkap', 'berjalan', 'menemukan', 'dinilai', 'dianggap', 'dirawat', 'menambahkan', 'menuntut', 'berdasarkan', 'meraih', 'menjalani', 'memilih', 'terlibat', 'gagal', 'ikut', 'menjelaskan', 'berasal', 'disampaikan', 'bertahan', 'dibawa', 'menunggu', 'merasa', 'menghadapi', 'bernama', 'mengambil', 'meninggal', 'bertemu', 'digunakan', 'membantu', 'mengikuti', 'tiba', 'menunjukkan', 'menegaskan', 'punya', 'diperiksa', 'berusaha', 'menyusul', 'mengetahui', 'berakhir', 'menyebutkan', 'mengakui', 'keluar', 'tampak', 'hadir', 'mengeluarkan', 'meninggalkan', 'membeli', 'mempunyai', 'pulang', 'memutuskan', 'menangkap', 'dikenal', 'ditahan', 'diberikan', 'berunjuk rasa', 'dilaporkan', 'diungkapkan', 'memenuhi', 'menyebabkan', 'diharapkan', 'memeriksa', 'dilansir', 'membuka', 'memastikan', 'menarik', 'menderita', 'terdapat', 'mendatangi', 'berjanji', 'membantah', 'mendesak', 'menjaga', 'meningkat', 'dijual', 'bergabung', 'muncul', 'terbukti', 'memasuki', 'mendukung', 'diterima', 'diambil', 'menyita', 'menjual', 'membayar', 'menanggapi', 'berupa', 'memperoleh', 'berarti', 'dimulai', 'menyerahkan', 'menahan', 'mencetak', 'mengungkapkan', 'menambah', 'dikutip', 'mengakibatkan', 'melaporkan', 'dibanding', 'mengingat', 'disebabkan', 'menyelesaikan', 'dipastikan', 'menangani', 'menurunkan', 'mengajukan', 'bertambah', 'selesai', 'mengancam', 'dijadikan', 'berbeda', 'melibatkan', 'terkena', 'menimbulkan', 'membutuhkan', 'berencana', 'mengantisipasi', 'membahas', 'menemui', 'menutup', 'dinyatakan', 'memimpin', 'diminta', 'meningkatkan', 'tercatat', 'menyampaikan', 'mencoba', 'melanggar', 'berisi', 'membangun', 'terbakar', 'lolos', 'diberi', 'dikatakan', 'menyerang', 'jatuh', 'mengungsi', 'tambah', 'dipimpin', 'menaikkan', 'terancam', 'berubah', 'menetapkan', 'berangkat', 'memanfaatkan', 'terendam', 'kehilangan', 'menghindari', 'ditetapkan', 'dialami']