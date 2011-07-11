#!/bin/sh

#HTML=all.html
#TXT=all.html
#mv -f $HTML $HTML.bak

#ALL=$@;

ALL=4chan 50_Cent Adrian_Lamo Afghan_War_documents_leak                  \
 Alternate_versions_of_Superman Alternative_hip_hop Apple_pie Aquablue   \
 Arts_and_Crafts_Movement A_Tribe_Called_Quest Auckland Audiosurf        \
 Avon_Barksdale Bayesian_inference Ben_Hana Big_Pun Boil Book            \
 Bradley_Manning Butter Cheese Cheryl_Cole                               \
 Citation Climate_change_policy_of_the_United_States Common_Lisp         \
 Cosmic_ordering Creation Damien_Hirst David_J._C._MacKay                \
 Doctor_Who_missing_episodes Doctor_Who Dr._Dre Drum_kit                 \
 Eating_disorder Eleventh_Doctor Encyclopedia_Dramatica                  \
 Environmental_impact_of_transport Extraterrestrial_hypothesis Fat_Joe   \
 Flash_mob Flight_of_the_Conchords Francis_Bacon FUBAR Geoffrey_Hinton   \
 Girls_Aloud Green_Day "Grendel's_Cave" Helen_Clark Hip_hop_music House  \
 Internet_meme Internet_Meme Internet Ireland Irish_republicanism        \
 Jane_McGonigal Jay-Z Jennifer_Aniston Jennifer_Lopez Joe_Klein          \
 John_Key Julian_Assange Justin_Bieber Karl_Marx Lady_Gaga               \
 Left-libertarianism Lip_sync LOL Meander Mila_Kunis Militant_atheism    \
 Milk Minister Nintendo NSFW Oil Onehunga Paint Parental_Advisory        \
 Pauley_Perrette Polynesians Prime Ragnarawk Remix Rock_music            \
 Role-playing_video_game Rosa_Parks Shweta_Menon Snoop_Dogg Soccer       \
 Social_entrepreneurship Star_Wars_Episode_V:_The_Empire_Strikes_Back    \
 Suburb Sweepstakes Taika_Waititi Taliban Te_Aro The_Phoenix_Foundation  \
 The_Wire Thomas_Bayes Todd_Solondz Transphobia Twitter UFO              \
 Uncyclopedia Video_game Wall WikiLeaks Wiki

for f in $ALL; do  
    curl -s "http://en.wikipedia.org/wiki/Talk:$f" | html2text  -utf8 -o $f-talk.txt
    curl -s "http://en.wikipedia.org/wiki/$f" | html2text  -utf8 -o $f.txt
done
