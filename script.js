document.addEventListener('DOMContentLoaded', (event) => {
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Simple typewriter effect for the main heading
    const typeWriter = (text, i, fnCallback) => {
        if (i < text.length) {
            document.querySelector("h2").innerHTML = text.substring(0, i+1) + '<span aria-hidden="true"></span>';
            setTimeout(() => {
                typeWriter(text, i + 1, fnCallback)
            }, 100);
        } else if (typeof fnCallback == 'function') {
            setTimeout(fnCallback, 700);
        }
    }
    // Start the typewriter effect
    typeWriter("Data Scientist | Machine Learning Engineer", 0);

    // Add a simple fade-in effect for project cards
    const projectCards = document.querySelectorAll('.project-card');
    projectCards.forEach((card, index) => {
        setTimeout(() => {
            card.style.opacity = 1;
        }, 200 * index);
    });
});
