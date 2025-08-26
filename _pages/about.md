---
permalink: /
title: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

<div style="text-align: justify; font-size: 90%;"> 
Hi, my name is Zican Wang, you can call me Robert. I am a PhD student from <a href="https://www.ucl.ac.uk" class="mylink">University College London</a>, supervised by Prof.<a href="https://www.homepages.ucl.ac.uk/~ucactri/" class="mylink">Tobias Ritschel</a> and Prof.<a href="http://www0.cs.ucl.ac.uk/staff/N.Mitra/" class="mylink">Niloy Mitra</a>.
I currently work on deep learning and visual computing related areas. 

</div>

<br/>
<p style="font-size: 35px; font-weight: 700;">Publications</p>
<ul>
  {% for post in site.publications reversed %}
    {% unless post.hidden == true %}
      {% include archive-single.html %}
    {% endunless %}
  {% endfor %}
</ul>