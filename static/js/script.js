// script.js - navigasi mobile, animasi statistik, drag & drop upload, dan render hasil analisis

// ===== Navbar toggle untuk tampilan mobile =====
const navToggle = document.getElementById("navToggle");
const navMenu = document.getElementById("navMenu");
const iconMenu = document.getElementById("iconMenu");
const iconClose = document.getElementById("iconClose");

if (navToggle) {
    navToggle.addEventListener("click", () => {
        navMenu.classList.toggle("active");
        const isActive = navMenu.classList.contains("active");
        iconMenu.style.display = isActive ? "none" : "block";
        iconClose.style.display = isActive ? "block" : "none";
    });
}

// ===== Animasi angka statistik di halaman beranda =====
const statNumbers = document.querySelectorAll(".stat-number");

if (statNumbers.length > 0) {
    const animateNumber = (el) => {
        const target = parseInt(el.dataset.target, 10);
        const suffix = el.dataset.suffix || "";
        const duration = 1200;
        const startTime = performance.now();

        const step = (now) => {
            const progress = Math.min((now - startTime) / duration, 1);
            const value = Math.floor(progress * target);
            el.innerText = value.toLocaleString("id-ID") + suffix;
            if (progress < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                animateNumber(entry.target);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });

    statNumbers.forEach((el) => observer.observe(el));
}

// ===== Logika halaman Klasifikasi =====
const uploadBox = document.getElementById("uploadBox");
const fileInput = document.getElementById("fileInput");
const previewImage = document.getElementById("previewImage");
const uploadText = document.getElementById("uploadText");
const uploadIcon = document.getElementById("uploadIcon");
const validationText = document.getElementById("validationText");
const analyzeBtn = document.getElementById("analyzeBtn");
const loadingBox = document.getElementById("loadingBox");
const progressFill = document.getElementById("progressFill");
const resultCard = document.getElementById("resultCard");
const resultLabel = document.getElementById("resultLabel");
const resultIcon = document.getElementById("resultIcon");
const resultBadge = document.getElementById("resultBadge");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceText = document.getElementById("confidenceText");
const detailMaterial = document.getElementById("detailMaterial");
const detailWaktu = document.getElementById("detailWaktu");
const detailStatus = document.getElementById("detailStatus");
const impactText = document.getElementById("impactText");
const handlingText = document.getElementById("handlingText");
const altList = document.getElementById("altList");

let selectedFile = null;

// Kumpulan ikon SVG per jenis material, dipilih berdasarkan icon_key dari backend
const ICONS = {
    paper: '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="3" width="14" height="18" rx="1"/><line x1="8" y1="8" x2="16" y2="8"/><line x1="8" y1="12" x2="16" y2="12"/><line x1="8" y1="16" x2="13" y2="16"/></svg>',
    cardboard: '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8 12 4 21 8 12 12 3 8Z"/><path d="M3 8 L3 17 L12 21 L21 17 L21 8"/><line x1="12" y1="12" x2="12" y2="21"/></svg>',
    biological: '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 4 13V7a1 1 0 0 1 1-1h6a7 7 0 0 1 7 7 7 7 0 0 1-7 7Z"/><path d="M11 20v-9"/><path d="M4 7c4 0 7 3 7 7"/></svg>',
    metal: '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="4" rx="1"/><rect x="4" y="10" width="16" height="4" rx="1"/><rect x="4" y="16" width="16" height="4" rx="1"/></svg>',
    plastic: '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="8" y="8" width="8" height="13" rx="2"/><rect x="10" y="3" width="4" height="5" rx="1"/><line x1="8" y1="13" x2="16" y2="13"/></svg>',
    glass: '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3 h8 l-1 8 a3 3 0 0 1 -6 0 L8 3 Z"/><line x1="12" y1="14" x2="12" y2="21"/><line x1="8" y1="21" x2="16" y2="21"/></svg>',
    clothes: '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 4 4 7 6 10 8 8 8 20 16 20 16 8 18 10 20 7 16 4 14 6 10 6 Z"/></svg>',
    shoes: '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 20 3 12 7 12 9 15 14 15 14 12 17 12 21 16 21 20 Z"/><line x1="3" y1="17" x2="21" y2="17"/></svg>',
    battery: '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="18" height="10" rx="2"/><line x1="22" y1="11" x2="22" y2="13"/></svg>',
    trash: '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>'
};

// Mengubah "brown-glass" menjadi "Brown Glass" untuk ditampilkan
function formatLabel(label) {
    return label
        .split("-")
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ");
}

if (uploadBox) {

    uploadBox.addEventListener("click", () => fileInput.click());

    uploadBox.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = "#3ED598";
    });

    uploadBox.addEventListener("dragleave", () => {
        uploadBox.style.borderColor = "#24392C";
    });

    uploadBox.addEventListener("drop", (e) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        handleFile(file);
    });

    fileInput.addEventListener("change", (e) => {
        handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        if (!file) return;

        if (!file.type.startsWith("image/")) {
            validationText.innerText = "File yang diunggah harus berupa gambar.";
            return;
        }

        validationText.innerText = "";
        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.style.display = "block";
            uploadText.style.display = "none";
            uploadIcon.style.display = "none";
        };
        reader.readAsDataURL(file);
    }

    analyzeBtn.addEventListener("click", () => {
        if (!selectedFile) {
            validationText.innerText = "Silakan pilih atau seret foto terlebih dahulu.";
            return;
        }

        resultCard.style.display = "none";
        loadingBox.style.display = "block";
        progressFill.style.width = "0%";

        let progress = 0;
        const progressInterval = setInterval(() => {
            progress = Math.min(progress + 10, 90);
            progressFill.style.width = progress + "%";
        }, 150);

        const formData = new FormData();
        formData.append("file", selectedFile);

        fetch("/predict", {
            method: "POST",
            body: formData
        })
        .then((response) => response.json())
        .then((data) => {
            clearInterval(progressInterval);
            progressFill.style.width = "100%";

            setTimeout(() => {
                loadingBox.style.display = "none";
                showResult(data);
            }, 300);
        })
        .catch(() => {
            clearInterval(progressInterval);
            loadingBox.style.display = "none";
            validationText.innerText = "Terjadi kesalahan saat menganalisis gambar.";
        });
    });

    function showResult(data) {
        const statusClass = data.is_recyclable ? "recyclable" : "residue";

        resultLabel.innerText = formatLabel(data.label);

        resultIcon.className = "result-icon " + statusClass;
        resultIcon.innerHTML = ICONS[data.icon_key] || ICONS.trash;

        resultBadge.className = "result-badge " + statusClass;
        resultBadge.innerText = data.is_recyclable ? "Dapat Didaur Ulang" : "Sampah Residu";

        confidenceFill.style.width = data.confidence + "%";
        confidenceText.innerText = "Tingkat keyakinan model: " + data.confidence + "%";

        detailMaterial.innerText = data.tipe_material;
        detailWaktu.innerText = data.waktu_terurai;
        detailStatus.innerText = data.is_recyclable ? "Dapat Didaur Ulang" : "Sampah Residu";

        impactText.innerText = data.dampak_lingkungan;
        handlingText.innerText = data.cara_penanganan;

        let altHtml = "";
        data.top3.forEach((item) => {
            if (item.label !== data.label) {
                altHtml += `
                    <div class="alt-row">
                        <span class="alt-name">${formatLabel(item.label)}</span>
                        <div class="alt-bar-track">
                            <div class="alt-bar-fill" style="width:${item.confidence}%;"></div>
                        </div>
                        <span class="alt-percent">${item.confidence}%</span>
                    </div>
                `;
            }
        });
        altList.innerHTML = altHtml;

        resultCard.style.display = "block";
        resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
}