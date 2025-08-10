document.addEventListener('DOMContentLoaded', function() {
        
    // 1. Fix ARIA issues in RTD search
    const searchInput = document.querySelector('.wy-side-nav-search input[type="text"]');
    if (searchInput) {
        searchInput.setAttribute('role', 'searchbox');
        searchInput.setAttribute('aria-label', 'Search docs');
    }
    
    // 2. Fix navigation ARIA labels
    const navElements = document.querySelectorAll('[role="navigation"]');
    navElements.forEach(nav => {
        if (!nav.hasAttribute('aria-label')) {
            nav.setAttribute('aria-label', 'Main navigation');
        }
    });
    
    // 3. Fix missing aria-level on role="heading" elements
    const headings = document.querySelectorAll('[role="heading"]:not([aria-level])');
    headings.forEach(function(heading) {
        heading.setAttribute('aria-level', '2');
    });
    
    // Video ID to Title mapping from your CSV
    const videoTitleMap = {
        'EnIKxTf8kiU': 'STAT 350 - Course Intro',
        'DcDqlxacmRY': 'STAT 350 - Chapter 1.1 Intro',
        'olkU2T4d8PI': 'STAT 350 - Intro to R',
        'PE0EBtI4ffk': 'STAT 350 - Chapter 1.2 Probability Population to Sample - Inference Sample to Population',
        'SKYjEnzY75I': 'STAT 350 - Chapter 2.1 Tables and Graphs for Summarizing Data - Structured Data Sets',
        'g8A7vIt8L9o': 'STAT 350 - Chapter 2.2 Graphical Summaries- Categorical-Qualitative Tools_1',
        'EiVdnnZtcRI': 'STAT 350 - Chapter 2.3 Graphical Summaries- Numerical-Quantitative Tools',
        'XwtyBLVThPY': 'STAT 350 - Chapter 2.4 Exploring Quantitative Distributions- Modality, Shape and Outliers',
        'SR-68DQX4Gs': 'STAT 350 - Chapter 3.1 Intro Numerical Summary Measures',
        'jWyWxBhBZZY': 'STAT 350 - Chapter 3.2 Measures of Central Tendency',
        'Bc85TbLQ11M': 'STAT 350 - Chapter 3.3 Measures of Variability-Spread',
        'ktc3R0fC4C8': 'STAT 350 - Chapter 3.4 Measures of Variability-Spread Inter Quartile Range',
        'if-8h2DECQg': 'STAT 350 - Chapter 3.5 Choosing Measures for Center and Spread_Resistant and Non-Resistant Measures',
        'xrMHgS064WE': 'STAT 350 - Chapter 4.1 Basic Set Theory',
        'PsanPPT3pW8': 'STAT 350 - Chapter 4.2 Probability',
        'JMZDN70PDO0': 'STAT 350 - Chapter 4.3 Conditional Probability',
        '4XDj9VRCVtE': 'STAT 350 - Chapter 4.4 Law of Total Probability and Bayes Rule',
        'KJK5tMOz89g': 'STAT 350 - Chapter 4.5 Bayes Update Rule Example',
        '_3Ukdl7pGPE': 'STAT 350 - Chapter 4.6 Independence of Events',
        'Inkj1RtLA_Q': 'STAT 350 - Chapter 5.1 Random Variables and Discrete Probability Distributions',
        'eJa8C_Yg0dg': 'STAT 350 - Chapter 5.2 Joint Probability Mass Function',
        'hTusBEM88fA': 'STAT 350 - Chapter 5.3 Expected Value of a Discrete Random Variable',
        'gA4f4mpjGk0': 'STAT 350 - Chapter 5.4 Spread of a Discrete Random Variable',
        'xP5_W5ZtBYs': 'STAT 350 - Chapter 5.5 Dependent Random Variables - Variance and Covariance',
        '1WON80Ut7lc': 'STAT 350 - Chapter 5.6 Special Case Discrete Probability Distributions Binomial Distribution',
        'L9flxu2RCEc': 'STAT 350 - Chapter 5.7 Special Case Discrete Probability Distributions Poisson Distribution',
        'F_crmr4FAcg': 'STAT 350 - Chapter 6.1 Continuous Random Variables',
        '_5PodnOjT5o': 'STAT 350 - Chapter 6.2 Expected Value and Variance',
        'cq_a1PFV0wQ': 'STAT 350 - Chapter 6.3 Cumulative Distribution Function',
        'hbpqL-h0830': 'STAT 350 - Chapter 6.3.1 Continuous Random Variable Example 1',
        'G-u5vHtQI3s': 'STAT 350 - Chapter 6.3.2 Continuous Random Variable Example 2',
        'O3wz4JgtZsA': 'STAT 350 - Chapter 6.4 Gaussian Normal Distribution',
        'IGnLAeROI44': 'STAT 350 - Chapter 6.4.1 Normal Distribution Probabilities - Forward Problems',
        'nExxuvoX-gQ': 'STAT 350 - Chapter 6.4.2 Normal Distribution Probabilities - Backward Problems',
        'iuWe6rxgNbI': 'STAT 350 - Chapter 6.4.3 Checking Normality of Data',
        'dJLgVD_ViHc': 'STAT 350 - Chapter 6.5 More Named Continuous Distributions - Uniform Distribution',
        'NM3PD-pO-94': 'STAT 350 - Chapter 6.6 More Named Continuous Distributions - Exponential Distribution',
        'YLCUV0h1mRA': 'STAT 350 - Chapter 7.1 Statistics Sampling Distributions',
        'TIOCX2hjXqw': 'STAT 350 - Chapter 7.2 Sampling Distribution for the Sample Mean',
        'l1vhy86sIVU': 'STAT 350 - Chapter 7.3 Central Limit Theorem CLT',
        'U98siSK61oY': 'STAT 350 - Chapter 7.4 Discret Random Variables and the CLT',
        '6iP17gg247k': 'STAT 350 - Chapter 8.1 Experimental and Sampling Designs',
        'BOFWktiCddI': 'STAT 350 - Chapter 8.2 Experimental Design Principles',
        'wOwmbU10sh4': 'STAT 350 - Chapter 8.3 Basic Types of Experimental Designs',
        'fPg-KKi9YKo': 'STAT 350 - Chapter 8.4 Experimental Design Issues',
        '-2i5Gn4FseQ': 'STAT 350 - Chapter 8.5 Design Examples',
        'kaUeguNY8mU': 'STAT 350 - Chapter 8.6 Sampling Design',
        '9LywzLqOHCY': 'STAT 350 - Chapter 8.7 Sampling Bias',
        'P3Nyg84h0A8': 'STAT 350 - Chapter 9.1 Single Sample Confidence Intervals',
        'siP0lHZSjn8': 'STAT 350 - Chapter 9.2 Precision of a Confidence Interval and Sample Size Calculation',
        'eK7cWzaG0-0': 'STAT 350 - Chapter 9.3 Confidence Intervals o-known',
        '7bF1fBzg1cQ': 'STAT 350 - Chapter 9.4 Confidence Bounds o-known',
        '3ZhAnYsmILo': 'STAT 350 - Chapter 9.5 Confidence Intervals when o is Unknown_Students\' t-distribution',
        'ZQusNqSNSdY': 'STAT 350 - Chapter 10.1 Hypothesis Testing for the Mean of a Population and Power',
        'rc1OOsAohSw': 'STAT 350 - Chapter 10.1.1 Type 1, Type 2 Error and Power',
        'pXRyQQt_v_I': 'STAT 350 - Chapter 10.1.2 Power Calculations',
        'umlrWPs7qlA': 'STAT 350 - Chapter 10.1.3 Sample Size Calculations',
        'vjzyQHJrHE0': 'STAT 350 - Chapter 10.2 Hypothesis Testing and Power for the Mean of a Population',
        'oVXZ-UAhrwQ': 'STAT 350 - Chapter 10.3 Hypothesis Test and Confidence Interval-Bound',
        'Qf1OChGzcQE': 'STAT 350 - Chapter 10.3.1 Test Statistic when o is Unknown',
        'igQdAxeXEr8': 'STAT 350 - Chapter 10.4 What Is A Test of Significance',
        'x8RSig7k-Xo': 'STAT 350 - Chapter 10.4.1 Statisticallt Significant But is it of Practical Significance',
        'bztTXSBCIVo': 'STAT 350 - Chapter 11.1 Condfidence Interval and Hypothesis Testing for Two Samples or Treatments',
        'OKJxoLTK9GY': 'STAT 350 - Capter 11.2 Comparing Two Population Means Using Independent Samples',
        'CuhPeUL6wEo': 'STAT 350 - Chapter 11.3 Comparing Two Population Means Using Independent Samples_Pooled Estimator',
        '875mJJL5hrQ': 'STAT 350 - Chapter 11.4 Comparing Two Population Means Using Independent Samples_No Equal Variance Assuption',
        'uzpqYPvmcYE': 'STAT 350 - Chapter 11.5 Only Using the Un-Pooled Estimator',
        '9qEfrrRcbRw': 'STAT 350 - Chapter 11.6 Comparing Two Population Means Using Paired Samples',
        'FYgP2E9lre4': 'STAT 350 - Chapter 12.1 One-Way Anova',
        'BKEQadpmPzw': 'STAT 350 - Chapter 12.2 One-Way ANOVA Model and the Sources of Variability',
        'wr-jFQm3DzM': 'STAT 350 - Chapter 12.3 One-Way Hypothesis Test and F-Test Statistic',
        '8hNoZPqspq0': 'STAT 350 - Chapter 12.4 One-Way ANOVA and Two Independent Sample t-test Relationship',
        '9BK1PxNtNjc': 'STAT 350 - Chapter 12.5 Multiple Comparison Procedures Family Wise Error Rates',
        'NfbWwbuUVEg': 'STAT 350 - Chapter 13.1 Correlation and Regression- Simple Linear Regression',
        '5wECoca89ls': 'STAT 350 - Chapter 13.2 Scatter Plots',
        'a9skiqjau8I': 'STAT 350 - Chapter 13.3 Simple Linear Regression Model',
        'nD9hKHIaUIQ': 'STAT 350 - Chapter 13.4 Simple Linear Regression ANOVA Table and Coefficient of Determination',
        'qSG28mV6fx4': 'STAT 350 - Chapter 13.5 Sample Pearson Correlation Coefficient',
        'p8kRL-vUpVo': 'STAT 350 - Chapter 13.6 Diagonostics for Model Assumptions',
        'mTQ3GU9rpys': 'STAT 350 - Chapter 13.7 Simple Linear Regression Model Inference - F-test',
        '_XCCR_oXcL0': 'STAT 350 - Chapter 13.8 Simple Linear Regression Model Inference - Slope and Intercept',
        'zUyxH0AL530': 'STAT 350 - Chapter 13.9 Prediction and Uncertainty - Confidence Intervals for the Mean Response at a Point_Prediction Intervals at a Point',
        'J8NtyRd48QU': 'STAT 350 - Chapter 13.10 Robustness to Normality Assumptions',
        'XiQd9bhOSl4': 'STAT 350 - Chapter 13.11 Linear Regression Prediction Example - Cetane Number'
    };
    
    // Add titles to all iframes that don't have them
    const iframes = document.querySelectorAll('iframe:not([title])');
    
    iframes.forEach(function(iframe, index) {
        const src = iframe.src;
        
        // Extract video ID from YouTube URL
        let videoId = null;
        
        // Handle different YouTube URL formats
        const patterns = [
            /embed\/([^?]+)/,          // embed format
            /watch\?v=([^&]+)/,        // watch format
            /youtu\.be\/([^?]+)/       // short format
        ];
        
        for (const pattern of patterns) {
            const match = src.match(pattern);
            if (match) {
                videoId = match[1];
                break;
            }
        }
        
        // Set the title based on video ID
        if (videoId && videoTitleMap[videoId]) {
            iframe.setAttribute('title', videoTitleMap[videoId]);
            console.log(`Added title "${videoTitleMap[videoId]}" to video ${videoId}`);
        } else {
            // Fallback for videos not in the map
            iframe.setAttribute('title', 'STAT 350 Video Content');
            console.warn(`No title found for video ID: ${videoId}`);
        }
    });
});