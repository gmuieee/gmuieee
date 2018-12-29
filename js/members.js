var members = {};

$(document).ready(function() {
    $.ajax(
        {
            url: "/data/members.json",
            success: function(result) {
                load2018Alumni(result['alums']);
                loadOfficers(result['officers']);
                // loadThirdYears(result[2020]);
                // loadSecondYears(result[2021]);
                $('#allMembersContainer').html(members['off']);
            }
        }
    );
})

function getHTMLString(person) {

    var classID = Math.floor(Math.random()*3);
    var className = ['green','red','blue'];
    var returnString = `<div class="member-container">
                            <div class="basic-details">
                                <img src="${person.img}" class="member-image">
                                <h3 class="member-name">${person.name}</h3>
								<h2 class ="member-status">${person.profile}</h2>
                            </div>
                            <div class="inner-details ${className[classID]}">
                                <div class="social-info">`;

    if (person.github_url!="") {
        returnString+=`<span><a href="${person.github_url}"><i id="githubLogo" class="fab fa-github textLogo"></i></a></span>`;
    }

    if (person.linkedin_url!="") {
        returnString+=`<span><a href="${person.linkedin_url}"><i id="linkedinLogo" class="fab fa-linkedin-in textLogo"></i></a></span>`;
    }

    if (person.additional_url!="") {
        returnString+=`<span><a href="${person.additional_url}"><i id="globeLogo" class="fas fa-globe"></i></a></span>`;
    }

    returnString+=`</div></div></div>`
    
    return returnString;
}

function loadOfficers(off) {
    // Populate Fourth Years
    members['off'] = off.map(officer => getHTMLString(officer)).join();
}

// function loadThirdYears(thirdYears) {
    // // Populate Third Years
    // members['thirdYears'] = thirdYears.map(thirdYear => getHTMLString(thirdYear)).join();
// }

// function loadSecondYears(secondYears) {
    // // Populate Second Years
    // members['secondYears'] = secondYears.map(secondYear => getHTMLString(secondYear)).join();
// }

function load2018Alumni(alum) {
    //  Populate 2018 alumnis
    members['alum'] = alum.map(alumni => getHTMLString(alumni)).join();
}

// function to highlight selected tab on batchButton
$("#batchButtons button").on("click", function() {
    var buttons = $('#batchButtons').children();
    for (button of buttons) {
        $(button).removeClass("active-batch");
    }
    $(this).addClass("active-batch");
});

// funtion to highlight selected tab on "allMembersButtons" Bar
$("#allMembersButtons button").on("click", function() {
    var buttons = $('#allMembersButtons').children();
    for (button of buttons) {
        $(button).removeClass("active-batch");
    }
    $(this).addClass("active-batch");
   
    // if "presentMembers" tab selected, make "all" as selected Tab on "batchButtons"
    if (this == buttons[0])
    {
        let allbuttons = $('#batchButtons').children();
        for (allbutton of allbuttons) {
            $(allbutton).removeClass("active-batch");
        }
        $(allbuttons[0]).addClass("active-batch");
    }

    //if "Alumni" tab selected, make "all" as selected Tab on "alumniButtons"
    else if (this == buttons[1])
    {
        let allbuttons = $('#alumniButtons').children();
        for (allbutton of allbuttons) {
            $(allbutton).removeClass("active-batch");
        }
        $(allbuttons[0]).addClass("active-batch");
    }
});

// function to highlight selected tab on alumniButtons Bar
$("#alumniButtons button").on("click", function() {
   var buttons = $("#alumniButtons").children();
   for (button of buttons) {
       $(button).removeClass("active-batch");
   }
   $(this).addClass("active-batch");
});


// Event Listeners
function clickAllPresentMembers() {
    $('#allMembersContainer').html(members['off']);
    //hide alumniButton Bar
    $("#alumniButtons").css("display", "none");
    //display present members batchButton
    $("#batchButtons").css("display", "inline-block");
}

// function clickFourthYears() {
    // $('#allMembersContainer').html(members['fourthYears']);
// }

// function clickThirdYears() {
    // $('#allMembersContainer').html(members['thirdYears']);
// }

// function clickSecondYears() {
    // $('#allMembersContainer').html(members['secondYears']);
// }

function clickAllAlumni() {
    clickAlumni();

    //hide present members batchButton
    $("#batchButtons").css("display", "none");
    //show alumniButton
    $("#alumniButtons").css("display", "inline-block");
}

function clickAlumni() {
    $('#allMembersContainer').html(members['alum']);
}