var context = {};

context.init = function() {
  // contents
  context.slides = slideshow.getSlides();
  context.namedSlides = {};

  for(var i=0; i<this.slides.length; ++i) {
    var slide = this.slides[i];
    if(slide.properties['name']) {
      var name = slide.properties['name'];
      this.namedSlides[name] = slide;
    }
  }
  if(console) console.log(this.namedSlides);
};

context.createTOCs = function() {
  var $navigator = $('<div class="remark-navigator"><i class="fa fa-bars"></i> </div>');
  var k = 0;
  for (var name in this.namedSlides) {
    //if (name == 'last-page') continue;
    var slide = this.namedSlides[name];
    var slideNo = slide.getSlideIndex();
    var icon;
    if (slide.properties['nav-marker']) {
      icon = slide.properties['nav-marker']; //String.fromCharCode(slide.properties['nav-marker']);
    }
    else { // (1) (2) (3) ...
      icon = String.fromCharCode(0x2474 + (k++)); // 0x2474: ⑴
    }

    var icon_html = '<a href="#'+name+'" data-slide-index="'+slideNo+'">'+icon+'</a>'
    var $icon = $(icon_html);
    $navigator.append($icon);
  }

  var slideIndex = 0;
  $(".remark-slide").each(function() {
    ++slideIndex;
    console.log(slideIndex);

    $nav = $navigator.clone();
    $(".remark-slide-content", $(this)).append($nav);

    var $a = $nav.find("a[data-slide-index]").filter(function() {
      return parseInt($(this).attr('data-slide-index')) < slideIndex;
    }).last();
    $a.css({"opacity": 1, "font-weight": "bold"});
  });
};

function slideCustomActions(slideshow) {
  context.init();
  context.createTOCs();

  /*
  if(window.twttr) {
    window.twttr.widgets.load();
  }
  */

  slideshow.on('showSlide', function (slide) {
    window.currentSlide = slide;
    var slideIndex = slide.getSlideIndex();
    window.$slide = $(".remark-slide-content:eq("+slideIndex+")");
    var $slide = window.$slide;

    /* execute custom script upon load */
    var $scripts = $slide.find("script[data-showslide]");

    if ($scripts.length > 0) {
      // execute after DOM has rendered
      $scripts.each( function(index, element) {
        console.log('Slide #' + (slideIndex+1) + ', Executing ' + element);
        eval(element.innerHTML);
      });
    }
  });
}
